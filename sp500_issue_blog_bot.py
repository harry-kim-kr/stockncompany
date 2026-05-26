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
ALLOWED_CATEGORIES = {"스마트팜", "미국증시", "기술", "일반"}
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
    failed_articles: int = 0
    success: bool = False
    errors: list[str] | None = None


def split_env_list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in re.split(r"[\n,]+", raw) if item.strip()]


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
    feed_urls = split_env_list("RSS_FEED_URLS")
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
다음 뉴스 기사 데이터를 기반으로 SEO 블로그 포스팅에 필요한 JSON만 생성하세요.

반드시 지켜야 할 규칙:
- 출력은 반드시 요구된 스키마를 따르는 JSON 객체여야 합니다.
- 모든 필드는 한글로 작성하세요.
- 기사 데이터에 없는 사실, 수치, 기업명, 인물명, 전망을 지어내지 마세요.
- 불확실한 내용은 단정하지 말고 확인 필요 또는 관전 포인트로 표현하세요.
- 유사 문서처럼 보이는 딱딱한 요약문 대신 자연스러운 문장형으로 작성하세요.
- category는 반드시 "스마트팜", "미국증시", "기술", "일반" 중 하나만 선택하세요.
- points는 정확히 3개로 작성하세요.

스키마:
{{
  "seo_title": "클릭을 유도하는 트렌디하고 검색량이 높은 제목 (의문형이나 감탄형 조합)",
  "hook_intro": "이 기사를 왜 읽어야 하는지 자연스럽게 설명하는 인간적인 도입부 문장 (2~3문장)",
  "summary": "기사 핵심 내용에 대한 친절한 2문장 요약",
  "points": ["핵심 포인트1", "핵심 포인트2", "핵심 포인트3"],
  "faq_question": "사람들이 이 이슈나 종목에 대해 지금 구글에 가장 많이 검색해볼 만한 질문 1가지",
  "faq_answer": "그 질문에 대한 명쾌하고 확실한 답변 2문장 (구글 스니펫 노출용)",
  "opinion": "개인 투자자가 주목해야 할 리스크 또는 기회 요인 (자산운용사 매니저 관점의 조언)",
  "category": "스마트팜 | 미국증시 | 기술 | 일반 중 택1"
}}

기사 제목: {article.title}
출처: {article.source_name}
URL: {article.url}
감지된 티커 후보: {detected_tickers}
RSS 요약: {article.rss_summary[:500]}
전처리된 본문: {article.clean_text}
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
                        "출력은 반드시 요구된 스키마를 따르는 JSON 형태여야 하며 한글로 작성하라. "
                        "근거 없는 추측, 존재하지 않는 수치, 출처에 없는 사실을 만들지 마세요."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("MAX_OPENAI_OUTPUT_TOKENS", "700")),
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
        "summary": str(payload.get("summary", "")).strip(),
        "points": points,
        "faq_question": str(payload.get("faq_question", "이 이슈에서 가장 중요한 점은 무엇인가요?")).strip(),
        "faq_answer": str(payload.get("faq_answer", "")).strip(),
        "opinion": str(payload.get("opinion", "")).strip(),
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

    return f"""
<h1>{html.escape(ai_payload["seo_title"])}</h1>
<p>👋 안녕하세요! 오늘 시장에서 가장 뜨거운 소식을 전해드립니다. {html.escape(ai_payload["hook_intro"])}</p>
<hr>
<blockquote style="background: #f9f9f9; border-left: 8px solid #007bff; padding: 15px; margin: 20px 0;">
  📌 <strong>핵심 3줄 요약 보기</strong><br>
  • {p0}<br>
  • {p1}<br>
  • {p2}
</blockquote>
<h2>🔍 이슈 핵심 파헤치기</h2>
<p>{html.escape(ai_payload["summary"])}</p>
<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<p><strong>Q. {html.escape(ai_payload["faq_question"])}</strong></p>
<p>A. {html.escape(ai_payload["faq_answer"])}</p>
<h2>💡 투자자 시선 &amp; 한 줄 인사이트</h2>
<p style="color: #2c3e50; font-weight: bold; background: #f0f7ff; padding: 10px; border-radius: 5px;">{html.escape(ai_payload["opinion"])}</p>
{related_html}
<p style="font-size: 0.9em; color: gray; margin-top: 30px;">원본 출처: <a href="{source_url}" target="_blank" rel="noopener noreferrer">{source_name}</a></p>
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
    return (
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
    metrics.success = metrics.uploaded_posts > 0 or metrics.failed_articles == 0
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

    seen_cache.add(article_cache_key(article))
    ai_payload = call_openai_for_article(article, metrics)
    if not ai_payload:
        metrics.failed_articles += 1
        return False

    tickers = detect_tickers(article, json.dumps(ai_payload, ensure_ascii=False))
    post_html = build_post_html(ai_payload, article, history, tickers)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        save_dry_run_output(ai_payload, post_html, article)
        metrics.uploaded_posts += 1
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
    logging.info("Uploaded Blogger post: %s", upload_result.get("url") or upload_result.get("id"))
    return True


def run_once() -> None:
    load_dotenv()
    if os.getenv("OPENAI_MODEL", OPENAI_MODEL) != OPENAI_MODEL:
        logging.warning("OPENAI_MODEL is fixed to %s for minimum-cost operation.", OPENAI_MODEL)

    metrics = RunMetrics(started_at=datetime.now(KST).isoformat(timespec="seconds"), errors=[])
    history = load_json_file(HISTORY_PATH, {"items": []})
    max_posts = int(os.getenv("MAX_POSTS_PER_RUN", "1"))
    seen_cache: set[tuple[str, str, str]] = set()

    try:
        if os.getenv("DRY_RUN", "false").lower() != "true" and already_posted_today(history):
            logging.info("A post has already been published today. Skipping.")
            return

        for entry, source_name in fetch_rss_entries():
            if metrics.uploaded_posts >= max_posts:
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
