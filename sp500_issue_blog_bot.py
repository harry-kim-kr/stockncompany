from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

import feedparser
import requests
import schedule
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import APIStatusError, OpenAI, RateLimitError


OPENAI_MODEL = "gpt-4o-mini"
KST_RUN_TIME = "07:00"
BLOGGER_SCOPES = ["https://www.googleapis.com/auth/blogger"]
ALLOWED_CATEGORIES = {"미국증시", "기술주", "재무분석", "일반"}
HISTORY_PATH = Path("history.json")
RUN_SUMMARY_PATH = Path("run_summary.json")
OUTPUT_DIR = Path("output")
KST = ZoneInfo("Asia/Seoul")
COMMON_TICKERS = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "TSLA",
    "AVGO",
    "JPM",
    "LLY",
    "V",
    "MA",
    "UNH",
    "XOM",
    "COST",
    "WMT",
    "NFLX",
    "PG",
    "JNJ",
    "HD",
    "KO",
    "BAC",
    "AMD",
    "INTC",
    "CRM",
    "ORCL",
    "ADBE",
    "PLTR",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


@dataclass
class NewsArticle:
    guid: str
    url: str
    title: str
    source_name: str
    published: str
    rss_summary: str
    clean_text: str


@dataclass
class RunMetrics:
    started_at: str
    finished_at: str | None = None
    processed_articles: int = 0
    skipped_duplicates: int = 0
    openai_calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    uploaded_posts: int = 0
    dry_run_outputs: int = 0
    failed_articles: int = 0
    skipped_thin_articles: int = 0
    blogger_urls: list[str] | None = None
    success: bool = False
    errors: list[str] | None = None


def split_env_list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in re.split(r"[\n,]+", raw) if item.strip()]


def get_rss_feed_urls() -> list[str]:
    raw = os.getenv("RSS_FEED_URLS", "").strip()
    if not raw:
        return []
    if "\n" in raw or ";" in raw:
        return [item.strip() for item in re.split(r"[\n;]+", raw) if item.strip()]
    return [raw]


def load_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("%s is invalid JSON. Starting with a clean structure.", path)
        return default


def save_json_file(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip().lower()


def make_history_sets(history: dict) -> tuple[set[str], set[str], set[str], set[str]]:
    items = history.get("items", [])
    guids = {item.get("guid", "") for item in items if item.get("guid")}
    urls = {normalize_url(item.get("url", "")) for item in items if item.get("url")}
    titles = {normalize_title(item.get("title", "")) for item in items if item.get("title")}
    dates = {item.get("published_date", "") for item in items if item.get("published_date")}
    return guids, urls, titles, dates


def is_duplicate_article(article: NewsArticle, history: dict) -> bool:
    guids, urls, titles, _ = make_history_sets(history)
    return (
        bool(article.guid and article.guid in guids)
        or bool(article.url and normalize_url(article.url) in urls)
        or bool(article.title and normalize_title(article.title) in titles)
    )


def article_cache_key(article: NewsArticle) -> tuple[str, str, str]:
    return (
        article.guid or "",
        normalize_url(article.url),
        normalize_title(article.title),
    )


def is_duplicate_in_run(article: NewsArticle, seen_cache: set[tuple[str, str, str]]) -> bool:
    guid, url, title = article_cache_key(article)
    return any(
        key
        for key in seen_cache
        if (guid and key[0] == guid) or (url and key[1] == url) or (title and key[2] == title)
    )


def already_posted_today(history: dict) -> bool:
    today = datetime.now(KST).strftime("%Y-%m-%d")
    return today in make_history_sets(history)[3]


def fetch_rss_entries() -> list[tuple[dict, str]]:
    feed_urls = get_rss_feed_urls()
    if not feed_urls:
        logging.warning("RSS_FEED_URLS is empty. Nothing to process.")
        return []

    entries: list[tuple[dict, str]] = []
    for feed_url in feed_urls:
        try:
            parsed = feedparser.parse(feed_url)
            source_name = parsed.feed.get("title", urlparse(feed_url).netloc)
            for entry in parsed.entries:
                entries.append((entry, source_name))
        except Exception as exc:
            logging.error("RSS fetch failed for %s: %s", feed_url, exc)
    return entries


def strip_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def remove_boilerplate(text: str) -> str:
    patterns = [
        r"관련\s*기사.*",
        r"추천\s*기사.*",
        r"ADVERTISEMENT.*",
        r"Advertisement.*",
        r"ad\s*choices.*",
        r"기자\s*[:：]?.*?@[\w\.-]+",
        r"[\w\.-]+@[\w\.-]+\.\w+",
        r"Copyright\s*©?.*",
        r"All rights reserved.*",
        r"무단전재.*",
        r"재배포\s*금지.*",
        r"저작권자.*",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?。！？다])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 20]


def compact_article_text(title: str, text: str) -> str:
    max_article_chars = int(os.getenv("MAX_ARTICLE_CHARS", "2500"))
    max_summary_chars = int(os.getenv("MAX_SUMMARY_INPUT_CHARS", "1200"))
    cleaned = remove_boilerplate(text)[:max_article_chars]
    sentences = split_sentences(cleaned)
    compacted = " ".join(sentences[:8]) if sentences else cleaned
    return compacted[:max_summary_chars].strip()


def is_thin_article(article: NewsArticle) -> bool:
    min_chars = int(os.getenv("MIN_CLEAN_ARTICLE_CHARS", "400"))
    text = article.clean_text.strip()
    if len(text) < min_chars:
        return True

    thin_patterns = [
        r"자세한\s*내용은\s*(링크|본문|원문)",
        r"more details?\s+at",
        r"read\s+more",
        r"continue\s+reading",
        r"full story",
        r"subscription required",
        r"내용\s*없음",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in thin_patterns)


def detect_tickers(article: NewsArticle | None = None, text: str = "") -> list[str]:
    source = f"{article.title if article else ''} {article.rss_summary if article else ''} {text}"
    matches = re.findall(r"\(([A-Z]{1,5})\)|\b([A-Z]{2,5})\b", source)
    flattened = {value for match in matches for value in match if value}
    return [ticker for ticker in sorted(flattened) if ticker in COMMON_TICKERS][:5]


def fetch_article_body(url: str) -> str:
    if not url:
        return ""
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; stockncompany-rss-bot/1.0)",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        timeout=20,
    )
    response.raise_for_status()
    return strip_html(response.text)


def entry_to_article(entry: dict, source_name: str) -> NewsArticle:
    url = entry.get("link", "")
    title = entry.get("title", "").strip()
    guid = entry.get("id") or entry.get("guid") or url or title
    rss_summary = strip_html(entry.get("summary", "") or entry.get("description", ""))
    published = entry.get("published", "") or entry.get("updated", "")

    try:
        body = fetch_article_body(url)
    except Exception as exc:
        logging.warning("Article body fetch failed, using RSS summary. url=%s error=%s", url, exc)
        body = rss_summary

    clean_text = compact_article_text(title, body or rss_summary)
    return NewsArticle(
        guid=guid,
        url=url,
        title=title,
        source_name=source_name,
        published=published,
        rss_summary=rss_summary,
        clean_text=clean_text,
    )


def build_openai_prompt(article: NewsArticle) -> str:
    detected_tickers = detect_tickers(article)
    return f"""
Create only one JSON object for a Korean SEO investment blog post.

Persona:
- Buy-side quantitative financial analyst and 10-year asset-management portfolio manager.
- Mission: identify where market expectations may be mispriced versus confirmed facts.
- Zero hallucination: never invent financial figures, consensus estimates, analyst targets, filings, company names, or dates not present in the source data.
- Noise-free: no greetings, no filler, no stale transitions such as "firstly" or "in summary".

Output rules:
- The JSON keys must exactly follow the schema below.
- All JSON values must be written in natural Korean for Korean readers.
- If the article does not provide enough data for FCF, capital allocation, macro, or moat analysis, say that the available article data is insufficient instead of inventing numbers.
- category must be exactly one of: "미국증시", "기술주", "재무분석", "일반".
- points must contain exactly 3 items.

Schema:
{{
  "seo_title": "클릭을 유도하는 트렌디하고 검색량이 높은 한국어 제목 (의문형이나 감탄형 조합)",
  "hook_intro": "이 이슈를 왜 지금 당장 주목해야 하는지 시장 기대치와 엮어 설명하는 강렬한 도입부 (2~3문장)",
  "expectation_gap": "현재 시장 컨센서스(기대치)와 이번에 발생한 실제 결과(실적/공시/뉴스) 사이의 가장 큰 차이점 분석 (Fact 중심)",
  "points": [
    "재무/이익 성장지표 관점의 핵심 포인트 1가지",
    "현금흐름(FCF) 품질 또는 자본배분(자사주/배당) 관점의 핵심 포인트 1가지",
    "거시경제(금리/환율) 및 산업 구조적 해자(Moat) 관점의 핵심 포인트 1가지"
  ],
  "faq_question": "개인 투자자들이 이 종목/이슈에 대해 현재 구글에 가장 많이 검색해볼 만한 핵심 질문 1가지",
  "faq_answer": "구글 추천 스니펫 상위 노출에 적합하도록 정제된 명쾌한 답변 2문장",
  "bull_vs_bear": {{
    "bull_thesis": "강한 매수/긍정 논리 1가지 (성장 모멘텀, Catalyst 중심)",
    "bear_thesis": "강한 반대/리스크 논리 1가지 (밸류에이션 부담, SBC 희석, 규제 등)"
  }},
  "market_mispricing": "시장이 현재 가장 크게 틀리거나 놓치고 있는 단 하나의 오판 포인트(Mispricing Point) 분석",
  "category": "미국증시 | 기술주 | 재무분석 | 일반 중 택1"
}}

Article title: {article.title}
Source: {article.source_name}
URL: {article.url}
Detected ticker candidates: {detected_tickers}
RSS summary: {article.rss_summary[:500]}
Cleaned article text: {article.clean_text}
""".strip()


def is_insufficient_quota_error(exc: APIStatusError) -> bool:
    try:
        body = exc.response.json()
    except Exception:
        return False
    error = body.get("error", {}) if isinstance(body, dict) else {}
    return error.get("code") in {"insufficient_quota", "rate_limit_exceeded"}


def call_openai_for_article(article: NewsArticle, metrics: RunMetrics) -> dict | None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = build_openai_prompt(article)
    metrics.openai_calls += 1
    metrics.input_chars += len(prompt)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You must output only one JSON object following the requested schema. "
                        "All JSON values must be written in natural Korean. "
                        "Do not fabricate facts, financial figures, estimates, analyst views, or sources."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_OPENAI_OUTPUT_TOKENS", "1500")),
            response_format={"type": "json_object"},
        )
    except RateLimitError as exc:
        message = "OpenAI quota or rate limit exceeded."
        logging.error("%s %s", message, exc)
        if metrics.errors is not None:
            metrics.errors.append(message)
        return None
    except Exception as exc:
        message = f"OpenAI request failed: {exc}"
        logging.error(message)
        if metrics.errors is not None:
            metrics.errors.append(message)
        return None

    content = response.choices[0].message.content or "{}"
    metrics.output_chars += len(content)
    if response.usage:
        metrics.input_tokens += response.usage.prompt_tokens or 0
        metrics.output_tokens += response.usage.completion_tokens or 0

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        message = f"JSON parsing error: {exc}"
        logging.error(message)
        if metrics.errors is not None:
            metrics.errors.append(message)
        return None

    return validate_ai_payload(payload)


def validate_ai_payload(payload: dict) -> dict:
    category = payload.get("category", "일반")
    if category not in ALLOWED_CATEGORIES:
        category = "일반"

    points = payload.get("points") or []
    points = [str(point).strip() for point in points if str(point).strip()][:3]
    while len(points) < 3:
        points.append("기사 원문에서 확인 가능한 추가 정보는 출처 링크를 통해 점검해보세요.")

    return {
        "seo_title": str(payload.get("seo_title", "오늘의 주요 뉴스, 지금 확인해야 할 핵심은?")).strip(),
        "hook_intro": str(payload.get("hook_intro", "")).strip(),
        "expectation_gap": str(payload.get("expectation_gap", "")).strip(),
        "points": points,
        "faq_question": str(payload.get("faq_question", "이 이슈에서 가장 중요한 점은 무엇인가요?")).strip(),
        "faq_answer": str(payload.get("faq_answer", "")).strip(),
        "bull_vs_bear": {
            "bull_thesis": str((payload.get("bull_vs_bear") or {}).get("bull_thesis", "")).strip(),
            "bear_thesis": str((payload.get("bull_vs_bear") or {}).get("bear_thesis", "")).strip(),
        },
        "market_mispricing": str(payload.get("market_mispricing", "")).strip(),
        "category": category,
    }


def find_related_posts(history: dict, ai_payload: dict, tickers: list[str]) -> list[dict]:
    items = list(reversed(history.get("items", [])))
    related = []
    ticker_set = set(tickers)
    for item in items:
        same_category = item.get("category") == ai_payload["category"]
        same_ticker = bool(ticker_set and ticker_set & set(item.get("tickers", [])))
        if not (same_category or same_ticker):
            continue
        title = item.get("ai_title") or item.get("title")
        url = item.get("blogger_url")
        if title and url:
            related.append({"title": title, "url": url})
        if len(related) >= int(os.getenv("MAX_RELATED_LINKS", "3")):
            break
    return related


def build_related_links_html(related_posts: list[dict]) -> str:
    if not related_posts:
        return ""
    items = "\n".join(
        f'  <li><a href="{html.escape(post["url"])}">{html.escape(post["title"])}</a></li>'
        for post in related_posts
    )
    return f"""
<h2>📚 함께 보면 좋은 글</h2>
<ul>
{items}
</ul>
""".strip()


def build_post_html(ai_payload: dict, article: NewsArticle, history: dict, tickers: list[str]) -> str:
    p0, p1, p2 = [html.escape(point) for point in ai_payload["points"][:3]]
    source_name = html.escape(article.source_name or "Original article")
    source_url = html.escape(article.url)
    related_html = build_related_links_html(find_related_posts(history, ai_payload, tickers))
    bull = html.escape(ai_payload["bull_vs_bear"]["bull_thesis"])
    bear = html.escape(ai_payload["bull_vs_bear"]["bear_thesis"])

    return f"""
<h1>{html.escape(ai_payload["seo_title"])}</h1>
<p>👋 {html.escape(ai_payload["hook_intro"])}</p>
<hr>
<blockquote style="background: #f9f9f9; border-left: 8px solid #007bff; padding: 15px; margin: 20px 0;">
  📌 <strong>기관 투자자 관점 핵심 3줄 요약</strong><br>
  • {p0}<br>
  • {p1}<br>
  • {p2}
</blockquote>
<h2>📊 시장 기대치와의 괴리 (Expectation Gap)</h2>
<p>{html.escape(ai_payload["expectation_gap"])}</p>
<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<p><strong>Q. {html.escape(ai_payload["faq_question"])}</strong></p>
<p>A. {html.escape(ai_payload["faq_answer"])}</p>
<h2>⚖️ 찬반 논리 점검 (Bull vs Bear Thesis)</h2>
<ul>
  <li><strong>상방 모멘텀 (Bull):</strong> {bull}</li>
  <li><strong>하방 리스크 (Bear):</strong> {bear}</li>
</ul>
<h2>💡 월가의 오판 포인트 (Market Mispricing)</h2>
<p style="color: #2c3e50; font-weight: bold; background: #f0f7ff; padding: 12px; border-radius: 5px;">{html.escape(ai_payload["market_mispricing"])}</p>
{related_html}
<p style="font-size: 0.9em; color: gray; margin-top: 30px;">원본 출처: <a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_name}</a></p>
<br>
<p style="font-size: 0.85em; color: #95a5a6; text-align: center;">※ 본 분석은 의사결정 참고용 데이터이며, 최종 투자 책임은 사용자 본인에게 있습니다.</p>
""".strip()


def get_blogger_service():
    credentials_path = os.getenv("GOOGLE_CLIENT_SECRET_FILE", "client_secret.json")
    token_path = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, BLOGGER_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
                raise RuntimeError(
                    "Google OAuth token is missing or invalid. Generate token.json locally, "
                    "then save its full JSON content as the GOOGLE_TOKEN_JSON GitHub secret."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, BLOGGER_SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("blogger", "v3", credentials=creds)


def blogger_labels(ai_payload: dict, tickers: list[str]) -> list[str]:
    labels = [ai_payload["category"]]
    labels.extend(tickers)
    deduped = []
    for label in labels:
        cleaned = re.sub(r"\s+", " ", str(label)).strip()[:80]
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def upload_to_blogger(ai_payload: dict, post_html: str, tickers: list[str]) -> dict:
    blog_id = os.environ["BLOGGER_BLOG_ID"]
    is_draft = os.getenv("PUBLISH_STATUS", "draft").lower() != "publish"
    body = {
        "kind": "blogger#post",
        "title": ai_payload["seo_title"],
        "content": post_html,
        "labels": blogger_labels(ai_payload, tickers),
    }

    service = get_blogger_service()
    result = (
        service.posts()
        .insert(
            blogId=blog_id,
            body=body,
            isDraft=is_draft,
            fetchImages=True,
            fetchBody=True,
        )
        .execute()
    )
    post_url = result.get("url")
    post_id = result.get("id")
    status = "draft" if is_draft else "published"
    logging.info("Blogger API accepted post. status=%s id=%s url=%s", status, post_id, post_url)
    if not post_id:
        raise RuntimeError(f"Blogger API response did not include a post id: {result}")
    return result


def save_dry_run_output(ai_payload: dict, post_html: str, article: NewsArticle) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^A-Za-z0-9가-힣_-]+", "_", ai_payload["seo_title"])[:60]
    base = OUTPUT_DIR / f"{timestamp}_{safe_title}"
    save_json_file(base.with_suffix(".json"), {"ai": ai_payload, "article": asdict(article)})
    base.with_suffix(".html").write_text(post_html, encoding="utf-8")
    logging.info("DRY_RUN output saved: %s.*", base)


def update_history(history: dict, article: NewsArticle, ai_payload: dict, upload_result: dict, tickers: list[str]) -> None:
    items = history.setdefault("items", [])
    items.append(
        {
            "guid": article.guid,
            "url": article.url,
            "title": article.title,
            "ai_title": ai_payload["seo_title"],
            "category": ai_payload["category"],
            "tickers": tickers,
            "published_date": datetime.now(KST).strftime("%Y-%m-%d"),
            "published_at": datetime.now(KST).isoformat(timespec="seconds"),
            "blogger_post_id": upload_result.get("id"),
            "blogger_url": upload_result.get("url"),
        }
    )
    keep = int(os.getenv("HISTORY_KEEP_ITEMS", "500"))
    history["items"] = items[-keep:]
    save_json_file(HISTORY_PATH, history)


def append_run_summary(metrics: RunMetrics) -> None:
    metrics.finished_at = datetime.now(KST).isoformat(timespec="seconds")
    metrics.success = metrics.uploaded_posts > 0 or metrics.dry_run_outputs > 0 or metrics.failed_articles == 0
    payload = load_json_file(RUN_SUMMARY_PATH, {"runs": []})
    payload.setdefault("runs", []).append(asdict(metrics))
    payload["runs"] = payload["runs"][-int(os.getenv("RUN_SUMMARY_KEEP_ITEMS", "200")) :]
    save_json_file(RUN_SUMMARY_PATH, payload)
    logging.info("Run summary: %s", json.dumps(asdict(metrics), ensure_ascii=False))


def process_one_article(
    article: NewsArticle,
    history: dict,
    metrics: RunMetrics,
    seen_cache: set[tuple[str, str, str]],
) -> bool:
    metrics.processed_articles += 1

    if is_duplicate_article(article, history) or is_duplicate_in_run(article, seen_cache):
        metrics.skipped_duplicates += 1
        logging.info("Skipping duplicate article: %s", article.title)
        return False

    if is_thin_article(article):
        metrics.skipped_thin_articles += 1
        logging.info("Skipping thin article before OpenAI call: %s", article.title)
        return False

    seen_cache.add(article_cache_key(article))
    ai_payload = call_openai_for_article(article, metrics)
    if not ai_payload:
        metrics.failed_articles += 1
        return False

    tickers = detect_tickers(article, json.dumps(ai_payload, ensure_ascii=False))
    post_html = build_post_html(ai_payload, article, history, tickers)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        save_dry_run_output(ai_payload, post_html, article)
        metrics.dry_run_outputs += 1
        return True

    try:
        upload_result = upload_to_blogger(ai_payload, post_html, tickers)
    except Exception as exc:
        metrics.failed_articles += 1
        message = f"Blogger upload failed: {exc}"
        logging.error(message)
        if metrics.errors is not None:
            metrics.errors.append(message)
        return False

    update_history(history, article, ai_payload, upload_result, tickers)
    metrics.uploaded_posts += 1
    if metrics.blogger_urls is not None and upload_result.get("url"):
        metrics.blogger_urls.append(upload_result["url"])
    logging.info("Uploaded Blogger post: %s", upload_result.get("url") or upload_result.get("id"))
    return True


def run_once() -> None:
    load_dotenv()
    if os.getenv("OPENAI_MODEL", OPENAI_MODEL) != OPENAI_MODEL:
        logging.warning("OPENAI_MODEL is fixed to %s for minimum-cost operation.", OPENAI_MODEL)

    metrics = RunMetrics(started_at=datetime.now(KST).isoformat(timespec="seconds"), errors=[], blogger_urls=[])
    history = load_json_file(HISTORY_PATH, {"items": []})
    max_posts = int(os.getenv("MAX_POSTS_PER_RUN", "1"))
    seen_cache: set[tuple[str, str, str]] = set()

    try:
        if os.getenv("DRY_RUN", "false").lower() != "true" and already_posted_today(history):
            logging.info("A post has already been published today. Skipping.")
            return

        for entry, source_name in fetch_rss_entries():
            if metrics.uploaded_posts + metrics.dry_run_outputs >= max_posts:
                break
            try:
                article = entry_to_article(entry, source_name)
                if not article.title or not article.url:
                    logging.info("Skipping article without title or URL.")
                    continue
                process_one_article(article, history, metrics, seen_cache)
            except Exception as exc:
                metrics.failed_articles += 1
                message = f"Article processing failed: {exc}"
                logging.error(message)
                if metrics.errors is not None:
                    metrics.errors.append(message)
                continue
    finally:
        append_run_summary(metrics)


def run_scheduler() -> None:
    load_dotenv()
    run_time = os.getenv("RUN_TIME_KST", KST_RUN_TIME)
    for weekday in (
        schedule.every().monday,
        schedule.every().tuesday,
        schedule.every().wednesday,
        schedule.every().thursday,
        schedule.every().friday,
    ):
        weekday.at(run_time).do(run_once)

    logging.info("Scheduler started. Weekday run time: %s KST, Monday-Friday", run_time)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    if os.getenv("RUN_NOW", "false").lower() == "true":
        run_once()
    else:
        run_scheduler()
