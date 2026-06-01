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

CASHFLOW_VALUE_KEYWORDS = [
    "cash-producing",
    "cash producing",
    "free cash flow",
    "fcf",
    "cash flow",
    "capital allocation",
    "shareholder return",
    "buyback",
    "share repurchase",
    "dividend",
    "cash-rich",
    "cash rich",
    "return on invested capital",
    "roic",
    "value stock",
    "quality stock",
]
CASHFLOW_VALUE_KEYWORDS_KO = [
    "현금흐름",
    "자유현금흐름",
    "자본배분",
    "자사주 매입",
    "배당",
    "주주환원",
    "가치주",
    "현금 창출",
    "투하자본수익률",
]
ETF_DIVIDEND_KEYWORDS = [
    "etf",
    "dividend yield",
    "distribution yield",
    "distribution",
    "nav erosion",
    "expense ratio",
    "income fund",
    "covered call",
    "총수익률",
    "배당 착시",
    "분배금",
    "기초자산",
]
AI_SEMICONDUCTOR_KEYWORDS = [
    "ai chip",
    "artificial intelligence chip",
    "gpu",
    "inference",
    "training",
    "data center capex",
    "semiconductor",
    "hbm",
    "memory chip",
    "asic",
    "accelerator",
    "nvidia",
    "amd",
    "intel ai",
    "chipmaker",
    "fab",
    "foundry",
]
OVERRIDE_AI_WITH_CASHFLOW = [
    "free cash flow",
    "cash-producing",
    "capital allocation",
    "cash-rich",
    "fcf",
    "buyback",
    "shareholder return",
]
LIMITED_SOURCE_PATTERNS = [
    "oops, something went wrong",
    "premium",
    "upgrade to read",
    "subscription",
    "sign in",
    "already have a subscription",
    "skip to navigation",
    "skip to main content",
    "skip to right column",
    "more from yahoo scout",
    "more from yahoo finance",
    "your privacy choices",
]
BOILERPLATE_PATTERNS = [
    "oops, something went wrong",
    "skip to navigation",
    "skip to main content",
    "skip to right column",
    "premium",
    "upgrade to read",
    "sign in",
    "terms and privacy policy",
    "your privacy choices",
    "more from yahoo",
    "advertisement",
    "sponsored",
    "cookie",
    "related articles",
]
ARTICLE_TYPE_LABELS = {
    "cashflow_value": "현금흐름/가치주 기사",
    "etf_dividend": "ETF/배당 기사",
    "ai_semiconductor": "AI/반도체 기사",
    "bigtech": "빅테크 기사",
    "consumer_cyclical": "소비재/경기민감주 기사",
    "general_market": "일반 시장 뉴스",
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
    manual_prompt_outputs: int = 0
    failed_articles: int = 0
    skipped_thin_articles: int = 0
    quota_or_rate_limit_blocked: bool = False
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


def has_premium_or_paywall_text(text: str) -> bool:
    patterns = [
        r"\bpremium\b",
        r"upgrade\s+to\s+read",
        r"subscription\s+required",
        r"subscribe\s+to\s+continue",
        r"sign\s+in\s+to\s+(read|continue|view)",
        r"register\s+to\s+(read|continue|view)",
        r"유료\s*기사",
        r"구독\s*회원",
        r"로그인\s*후",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def content_quality_score(article: NewsArticle) -> int:
    text = article.clean_text.strip()
    sentences = split_sentences(text)
    score = 0
    score += min(len(text) // 20, 40)
    score += min(len(sentences) * 5, 30)
    if re.search(r"\b\d+(\.\d+)?%?\b|\$\d+|billion|million|EPS|revenue|margin|cash flow|FCF", text, re.I):
        score += 15
    if re.search(r"company|market|sales|demand|chip|AI|cloud|margin|growth|guidance|earnings", text, re.I):
        score += 10
    if has_premium_or_paywall_text(f"{article.title} {article.rss_summary} {text}"):
        score -= 25
    if re.search(r"advertisement|related articles|read more|continue reading|all rights reserved", text, re.I):
        score -= 15
    return max(0, min(score, 100))


def is_limited_source(article: NewsArticle) -> bool:
    text = article.clean_text.strip()
    combined = f"{article.title} {article.rss_summary} {text}"
    meaningful_sentences = split_sentences(text)
    has_financial_or_event_detail = bool(
        re.search(
            r"\b(revenue|earnings|EPS|margin|cash flow|FCF|guidance|acquisition|lawsuit|export|sales|demand|supply|chip|AI|cloud|dividend|buyback)\b|\d",
            combined,
            re.I,
        )
    )
    return (
        len(text) < int(os.getenv("LIMITED_SOURCE_CHARS", "800"))
        or has_premium_or_paywall_text(combined)
        or len(meaningful_sentences) < int(os.getenv("MIN_MEANINGFUL_SENTENCES", "4"))
        or content_quality_score(article) < int(os.getenv("MIN_CONTENT_QUALITY_SCORE", "45"))
        or not has_financial_or_event_detail
    )


def classify_article_type(article: NewsArticle, tickers: list[str]) -> str:
    text = f"{article.title} {article.rss_summary} {article.clean_text} {' '.join(tickers)}".lower()
    if re.search(r"ai|chip|semiconductor|gpu|asic|hbm|memory|nvidia|amd|intel|data center|datacenter", text):
        return "AI/반도체 기사"
    if re.search(r"fcf|free cash flow|cash flow|roic|buyback|dividend|capital allocation|value stock", text):
        return "현금흐름/가치주 기사"
    if re.search(r"consumer|retail|inventory|brand|pricing power|discretionary|staples|sales slowdown", text):
        return "소비재/경기민감주 기사"
    if re.search(r"big tech|cloud|advertising|platform|regulation|capex|apple|microsoft|google|meta|amazon", text):
        return "빅테크 기사"
    if re.search(r"etf|dividend yield|distribution|nav|expense ratio|income fund", text):
        return "ETF/배당 기사"
    return "일반 시장/기업 뉴스"


def article_type_guide(article_type: str) -> str:
    guides = {
        "AI/반도체 기사": "- AI 학습 vs AI 추론\n- GPU, ASIC, 메모리, HBM, 저가 메모리\n- 엔비디아, AMD, 인텔, 빅테크 자체 칩\n- 공급망, 전력, 데이터센터 CAPEX\n- 성능보다 TCO가 중요한 구간",
        "현금흐름/가치주 기사": "- FCF\n- FCF Margin\n- ROIC\n- 자본배분\n- 자사주 매입\n- 배당\n- M&A\n- 현금흐름은 좋지만 성장성이 부족한 기업의 함정",
        "소비재/경기민감주 기사": "- 소비 사이클\n- 재고 부담\n- 금리 민감도\n- 수요 둔화\n- 브랜드 파워\n- 가격 전가력",
        "빅테크 기사": "- 클라우드\n- 광고\n- AI CAPEX\n- 플랫폼 락인\n- 규제 리스크\n- 주주환원",
        "ETF/배당 기사": "- 총수익률\n- 배당 착시\n- NAV 훼손\n- 비용률\n- 기초자산 리스크\n- 분배금 지속 가능성",
    }
    return guides.get(article_type, "- 기사에 확인된 사건\n- 산업 구조\n- 경쟁 구도\n- 수요와 비용 변수\n- 투자자가 확인해야 할 리스크")


def keyword_in_text(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def has_premium_or_paywall_text(text: str) -> bool:
    return keyword_in_text(text, LIMITED_SOURCE_PATTERNS)


def boilerplate_ratio(text: str) -> float:
    if not text.strip():
        return 1.0
    lowered = text.lower()
    boilerplate_chars = 0
    for pattern in BOILERPLATE_PATTERNS:
        boilerplate_chars += sum(len(match.group(0)) for match in re.finditer(re.escape(pattern), lowered))
    return min(1.0, boilerplate_chars / max(len(text), 1))


def article_body_ratio(text: str) -> float:
    sentences = split_sentences(text)
    if not sentences:
        return 0.0
    question_count = sum(1 for sentence in sentences if sentence.endswith("?"))
    boilerplate_sentence_count = sum(
        1 for sentence in sentences if keyword_in_text(sentence, BOILERPLATE_PATTERNS)
    )
    body_sentence_count = max(0, len(sentences) - question_count - boilerplate_sentence_count)
    return body_sentence_count / max(len(sentences), 1)


def content_quality_score(article: NewsArticle) -> int:
    text = article.clean_text.strip()
    sentences = split_sentences(text)
    paragraphs = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s{2,}", text) if len(part.strip()) >= 40]
    combined = f"{article.title} {article.rss_summary} {text}"
    boilerplate = boilerplate_ratio(text)
    body_ratio = article_body_ratio(text)

    score = 20
    score += min(len(text) // 120, 25)
    score += min(len(sentences) * 3, 25)
    score += min(len(paragraphs) * 3, 15)
    if re.search(r"\b\d+(\.\d+)?%?\b|\$\d+|billion|million|EPS|revenue|margin|cash flow|FCF", combined, re.I):
        score += 10
    if re.search(
        r"company|market|sales|demand|growth|guidance|earnings|cash flow|capital allocation|dividend|buyback|acquisition|lawsuit|export",
        combined,
        re.I,
    ):
        score += 10

    if has_premium_or_paywall_text(combined):
        score -= 30
    if keyword_in_text(text, BOILERPLATE_PATTERNS):
        score -= 20
    if len(text) < 3000:
        score -= 15
    if len(sentences) < 10:
        score -= 20
    if boilerplate > 0.35:
        score -= 25
    if body_ratio < 0.5:
        score -= 20
    question_only_count = sum(1 for sentence in sentences if sentence.endswith("?"))
    if question_only_count >= max(2, len(sentences) // 2):
        score -= 15

    if keyword_in_text(text, BOILERPLATE_PATTERNS) and len(text) >= 80:
        score = max(score, 30)

    return max(0, min(score, 100))


def is_limited_source(article: NewsArticle) -> bool:
    text = article.clean_text.strip()
    combined = f"{article.title} {article.rss_summary} {text}"
    meaningful_sentences = split_sentences(text)
    has_financial_or_event_detail = bool(
        re.search(
            r"\b(revenue|earnings|EPS|margin|cash flow|FCF|guidance|acquisition|lawsuit|export|sales|demand|supply|chip|AI|cloud|dividend|buyback)\b|\d",
            combined,
            re.I,
        )
    )
    return (
        len(text) < int(os.getenv("LIMITED_SOURCE_CHARS", "800"))
        or len(text) < 3000
        or has_premium_or_paywall_text(combined)
        or len(meaningful_sentences) < int(os.getenv("MIN_MEANINGFUL_SENTENCES", "4"))
        or len(meaningful_sentences) < 10
        or boilerplate_ratio(text) > 0.35
        or article_body_ratio(text) < 0.5
        or content_quality_score(article) < int(os.getenv("MIN_CONTENT_QUALITY_SCORE", "45"))
        or not has_financial_or_event_detail
    )


def classify_article_type(article: NewsArticle, tickers: list[str]) -> str:
    text = f"{article.title} {article.rss_summary} {article.clean_text} {' '.join(tickers)}".lower()
    strong_cashflow_keywords = [
        keyword for keyword in CASHFLOW_VALUE_KEYWORDS + CASHFLOW_VALUE_KEYWORDS_KO
        if keyword.lower() not in {"dividend", "배당"}
    ]
    if keyword_in_text(text, strong_cashflow_keywords):
        return "cashflow_value"
    if keyword_in_text(text, ETF_DIVIDEND_KEYWORDS):
        return "etf_dividend"
    if keyword_in_text(text, CASHFLOW_VALUE_KEYWORDS + CASHFLOW_VALUE_KEYWORDS_KO):
        return "cashflow_value"
    if keyword_in_text(text, AI_SEMICONDUCTOR_KEYWORDS) and not keyword_in_text(text, OVERRIDE_AI_WITH_CASHFLOW):
        return "ai_semiconductor"
    if re.search(r"big tech|cloud|advertising|platform|regulation|capex|apple|microsoft|google|meta|amazon", text):
        return "bigtech"
    if re.search(r"consumer|retail|inventory|brand|pricing power|discretionary|staples|sales slowdown", text):
        return "consumer_cyclical"
    return "general_market"


def article_type_guide(article_type: str) -> str:
    guides = {
        "cashflow_value": "- 자동 추정 기사 유형: 현금흐름/가치주 기사\n- 자유현금흐름(FCF)\n- FCF Margin\n- ROIC\n- 자본배분\n- 자사주 매입\n- 배당\n- M&A\n- 현금흐름은 좋지만 성장성이 부족한 기업의 함정\n- \"현금을 많이 버는 기업\"과 \"좋은 투자 대상\"의 차이",
        "ai_semiconductor": "- 자동 추정 기사 유형: AI/반도체 기사\n- AI 학습 vs AI 추론\n- GPU, ASIC, 메모리, HBM, 저가 메모리\n- 엔비디아, AMD, 인텔, 빅테크 자체 칩\n- 공급망, 전력, 데이터센터 CAPEX\n- 성능보다 TCO가 중요한 구간",
        "etf_dividend": "- 자동 추정 기사 유형: ETF/배당 기사\n- 총수익률\n- 배당 착시\n- NAV 훼손\n- 비용률\n- 기초자산 리스크\n- 분배금 지속 가능성",
        "consumer_cyclical": "- 자동 추정 기사 유형: 소비재/경기민감주 기사\n- 소비 사이클\n- 재고 부담\n- 금리 민감도\n- 수요 둔화\n- 브랜드 파워\n- 가격 전가력",
        "bigtech": "- 자동 추정 기사 유형: 빅테크 기사\n- 클라우드\n- 광고\n- AI CAPEX\n- 플랫폼 락인\n- 규제 리스크\n- 주주환원",
    }
    return guides.get(article_type, "- 자동 추정 기사 유형: 일반 시장 뉴스\n- 기사에 확인된 사건\n- 산업 구조\n- 경쟁 구도\n- 수요와 비용 변수\n- 투자자가 확인해야 할 리스크")


def article_type_label(article_type: str) -> str:
    return ARTICLE_TYPE_LABELS.get(article_type, ARTICLE_TYPE_LABELS["general_market"])


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
        metrics.quota_or_rate_limit_blocked = True
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
<p style="font-size: 0.9em; color: gray; margin-top: 30px;">원본 출처: <a href="{source_url}" target="_blank" rel="noopener noreferrer nofollow">{source_name}</a></p>
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


def build_manual_gpt_prompt(article: NewsArticle, history: dict, tickers: list[str]) -> str:
    related_posts = find_related_posts(
        history,
        {"category": "미국증시"},
        tickers,
    )
    related_text = "\n".join(
        f"- {post['title']}: {post['url']}" for post in related_posts
    ) or "- 관련 과거 글 없음"
    limited_source = is_limited_source(article)
    quality_score = content_quality_score(article)
    article_type = classify_article_type(article, tickers)
    guide = article_type_guide(article_type)
    limited_source_instruction = (
        """- 제한적 원문 여부: True
- 원문이 Premium/부분 공개 기사이거나 본문 데이터가 부족합니다.
- 얇은 기사 요약문으로 작성하지 마세요.
- 제목과 공개 요약을 "시드 이슈"로만 사용하고, 검증 가능한 일반 산업 지식 중심의 SEO 전략 분석 글로 확장하세요.
- 기사에 없는 최신 수치, 성능 데이터, 매출, EPS, 목표주가, 컨센서스, 애널리스트 의견은 절대 만들지 마세요.
- 필요한 경우 "기사 데이터만으로는 확인이 제한됩니다"라고 명시하세요."""
        if limited_source
        else
        """- 제한적 원문 여부: False
- 원문 본문이 충분하더라도 기사에 없는 재무 수치, 목표주가, 컨센서스, 공시 내용은 만들지 마세요.
- 단순 요약이 아니라 투자자 관점의 시장 기대치와 mispricing 가능성을 분석하세요."""
    )

    return f"""
당신은 기관투자자(Buy-side) 스타일의 퀀트 기반 재무분석가이자 10년 차 자산운용사 매니저입니다.

목표:
- 아래 기사 데이터를 바탕으로 구글 검색 상위 노출을 노리는 한국어 Blogger/Tistory용 HTML 글을 작성하세요.
- 단순 요약이 아니라 시장 기대치 대비 mispricing이 발생한 투자 기회와 위험을 분석하세요.
- 수집되지 않은 재무 수치, 컨센서스, 목표주가, 애널리스트 의견, 공시 내용은 절대 지어내지 마세요.
- 자료에 없는 내용은 "기사 데이터만으로는 확인이 제한됩니다"라고 표현하세요.
- 인사말, 감탄사, "첫째로", "요약하자면" 같은 진부한 표현 없이 바로 분석으로 들어가세요.

제한적 원문 처리:
{limited_source_instruction}
- 글 길이는 공백 제외 최소 2,000자 이상, 가능하면 2,500~3,500자 수준으로 작성하세요.

SEO 제목 작성 규칙:
- 검색자가 실제로 입력할 만한 키워드를 포함하세요.
- 단순 기사 제목 번역을 피하세요.
- 산업 구조, 투자 포인트, 경쟁 구도, 리스크를 반영하세요.
- 예:
  - "인텔 AI 반도체 재도전, 엔비디아 독주를 흔들 수 있을까?"
  - "현금흐름이 좋은 기업이 반드시 좋은 투자일까? FCF와 자본배분의 함정"
  - "AI 추론 시장이 커질수록 주목해야 할 반도체 투자 포인트"

기사 유형별 확장 가이드:
- 자동 추정 기사 유형: {article_type}
{guide}

출력:
- Blogger/Tistory에 바로 붙여넣을 수 있는 HTML만 출력하세요.
- Markdown 설명, 코드블록, 별도 해설은 출력하지 마세요.

HTML 구조:
<h1>SEO 제목</h1>
<p>이 이슈를 왜 지금 봐야 하는지 시장 기대치와 엮은 강한 도입부 2~3문장</p>
<hr>
<blockquote style="background: #f9f9f9; border-left: 8px solid #007bff; padding: 15px; margin: 20px 0;">
  📌 <strong>기관 투자자 관점 핵심 3줄 요약</strong><br>
  • 핵심 포인트 1<br>
  • 핵심 포인트 2<br>
  • 핵심 포인트 3
</blockquote>
<h2>📊 시장 기대치와의 괴리 (Expectation Gap)</h2>
<p>시장 기대치와 실제 뉴스 사이의 차이를 Fact 중심으로 분석</p>
<h2>🏢 기업의 현재 위치와 경쟁 구도</h2>
<p>분석 대상 기업이 현재 산업 안에서 어떤 위치에 있는지, 엔비디아/AMD/빅테크 등 관련 경쟁 구도와 함께 설명</p>
<h2>🧠 기술 전략과 원가 구조의 의미</h2>
<p>AI 추론, 메모리, 칩 설계, 공급망, 비용 구조 등 공개적으로 확인 가능한 산업 논리 중심 분석</p>
<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<p><strong>Q. 개인 투자자가 검색할 만한 핵심 질문</strong></p>
<p>A. 구글 추천 스니펫에 적합한 2문장 답변</p>
<h2>⚖️ 찬반 논리 점검 (Bull vs Bear Thesis)</h2>
<ul>
  <li><strong>상방 모멘텀 (Bull):</strong> 성장 모멘텀 또는 catalyst 중심 긍정 논리</li>
  <li><strong>하방 리스크 (Bear):</strong> 밸류에이션, 희석, 규제, 수요 둔화 등 반대 논리</li>
</ul>
<h2>💡 월가의 오판 포인트 (Market Mispricing)</h2>
<p style="color: #2c3e50; font-weight: bold; background: #f0f7ff; padding: 12px; border-radius: 5px;">시장이 놓치고 있을 수 있는 단 하나의 오판 포인트</p>
<h2>📚 함께 보면 좋은 글</h2>
<ul>
  관련 글이 있으면 관련 과거 글 목록을 활용해 자연스러운 앵커 링크 삽입
  관련 과거 글이 없으면 <li>관련 과거 글은 아직 없습니다. 향후 반도체, AI, 자유현금흐름, 배당 ETF 관련 분석 글을 추가할 예정입니다.</li> 사용
</ul>
<p style="font-size: 0.9em; color: gray; margin-top: 30px;">원본 출처: <a href="{html.escape(article.url)}" target="_blank" rel="noopener noreferrer">{html.escape(article.source_name)}</a></p>
<br>
<p style="font-size: 0.85em; color: #95a5a6; text-align: center;">※ 본 분석은 의사결정 참고용 데이터이며, 최종 투자 책임은 사용자 본인에게 있습니다.</p>

기사 데이터:
- 제목: {article.title}
- 출처: {article.source_name}
- URL: {article.url}
- 감지된 티커 후보: {tickers}
- RSS 요약: {article.rss_summary[:700]}
- 제한적 원문 여부: {limited_source}
- 본문 품질 점수: {quality_score}
- 기사 유형: {article_type}
- 전처리된 본문: {article.clean_text}

관련 과거 글:
{related_text}
""".strip()


def build_manual_gpt_prompt(article: NewsArticle, history: dict, tickers: list[str]) -> str:
    related_posts = find_related_posts(history, {"category": "미국증시"}, tickers)
    related_text = "\n".join(
        f"- {post['title']}: {post['url']}" for post in related_posts
    ) or "- 관련 과거 글 없음"
    limited_source = is_limited_source(article)
    quality_score = content_quality_score(article)
    article_type = classify_article_type(article, tickers)
    article_type_name = article_type_label(article_type)
    guide = article_type_guide(article_type)
    limited_source_instruction = (
        """- 제한적 원문 여부: True
- 원문이 Premium/부분 공개 기사이거나 본문 데이터가 부족합니다.
- 얇은 기사 요약문으로 작성하지 마세요.
- 제목과 공개 요약을 "시드 이슈"로만 사용하고, 검증 가능한 일반 산업 지식 중심의 SEO 전략 분석 글로 확장하세요.
- 기사에 없는 최신 수치, 성능 데이터, 매출, EPS, 목표주가, 컨센서스, 애널리스트 의견은 절대 만들지 마세요.
- 필요한 경우 "기사 데이터만으로는 확인이 제한됩니다"라고 명시하세요."""
        if limited_source
        else
        """- 제한적 원문 여부: False
- 원문 본문이 충분하더라도 기사에 없는 재무 수치, 목표주가, 컨센서스, 공시 내용은 만들지 마세요.
- 단순 요약이 아니라 투자자 관점의 시장 기대치와 mispricing 가능성을 분석하세요."""
    )

    fallback_related_html = (
        "<li>관련 과거 글은 아직 없습니다. 향후 반도체, AI, 자유현금흐름, "
        "배당 ETF 관련 분석 글을 추가할 예정입니다.</li>"
    )

    return f"""
당신은 기관투자자(Buy-side) 스타일의 퀀트 기반 재무분석가이자 10년 차 자산운용사 매니저입니다.

목표:
- 아래 기사 데이터를 바탕으로 구글 검색 상위 노출을 노리는 한국어 Blogger/Tistory용 HTML 글을 작성하세요.
- 단순 요약이 아니라 시장 기대치 대비 mispricing이 발생한 투자 기회와 위험을 분석하세요.
- 수집되지 않은 재무 수치, 컨센서스, 목표주가, 애널리스트 의견, 공시 내용은 절대 지어내지 마세요.
- 자료에 없는 내용은 "기사 데이터만으로는 확인이 제한됩니다"라고 표현하세요.
- 인사말, 감탄사, "첫째로", "요약하자면" 같은 진부한 표현 없이 바로 분석으로 들어가세요.

제한적 원문 처리:
{limited_source_instruction}
- 원문이 Premium/부분 공개 기사이거나 본문 데이터가 부족하면, 얇은 기사 요약문으로 쓰지 마세요.
- 이 경우 제목과 공개 요약을 "시드 이슈"로만 사용하고, 검증 가능한 일반 산업 지식 중심의 SEO 전략 분석 글로 확장하세요.
- 글 길이는 공백 제외 최소 2,000자 이상, 가능하면 2,500~3,500자 수준으로 작성하세요.

SEO 제목 작성 규칙:
- 검색자가 실제로 입력할 만한 키워드를 포함하세요.
- 단순 기사 제목 번역을 피하세요.
- 산업 구조, 투자 포인트, 경쟁 구도, 리스크를 반영하세요.
- 예:
  - "인텔 AI 반도체 재도전, 엔비디아 독주를 흔들 수 있을까?"
  - "현금흐름이 좋은 기업이 반드시 좋은 투자일까? FCF와 자본배분의 함정"
  - "AI 추론 시장이 커질수록 주목해야 할 반도체 투자 포인트"

기사 유형별 확장 가이드:
{guide}

출력:
- Blogger/Tistory에 바로 붙여넣을 수 있는 HTML만 출력하세요.
- Markdown 설명, 코드블록, 별도 해설은 출력하지 마세요.

HTML 구조:
<h1>SEO 제목</h1>

<p>이 이슈를 왜 지금 봐야 하는지 시장 기대치와 엮은 강한 도입부 2~3문장</p>

<hr>

<blockquote style="background: #f9f9f9; border-left: 8px solid #007bff; padding: 15px; margin: 20px 0;">
  📌 <strong>기관 투자자 관점 핵심 3줄 요약</strong><br>
  • 핵심 포인트 1<br>
  • 핵심 포인트 2<br>
  • 핵심 포인트 3
</blockquote>

<h2>📊 시장 기대치와의 괴리 (Expectation Gap)</h2>
<p>시장 기대치와 실제 뉴스 사이의 차이를 Fact 중심으로 분석</p>

<h2>🏢 기업의 현재 위치와 경쟁 구도</h2>
<p>분석 대상 기업이 현재 산업 안에서 어떤 위치에 있는지, 관련 경쟁 구도와 함께 설명</p>

<h2>🧠 기술 전략과 원가 구조의 의미</h2>
<p>AI 추론, 메모리, 칩 설계, 공급망, 비용 구조, 현금흐름, 자본배분 등 기사 유형에 맞는 공개 산업 논리 중심 분석</p>

<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<p><strong>Q. 개인 투자자가 검색할 만한 핵심 질문</strong></p>
<p>A. 구글 추천 스니펫에 적합한 2문장 답변</p>

<h2>⚖️ 찬반 논리 점검 (Bull vs Bear Thesis)</h2>
<ul>
  <li><strong>상방 모멘텀 (Bull):</strong> 성장 모멘텀 또는 catalyst 중심 긍정 논리</li>
  <li><strong>하방 리스크 (Bear):</strong> 밸류에이션, 희석, 규제, 수요 둔화 등 반대 논리</li>
</ul>

<h2>💡 월가의 오판 포인트 (Market Mispricing)</h2>
<p style="color: #2c3e50; font-weight: bold; background: #f0f7ff; padding: 12px; border-radius: 5px;">시장이 놓치고 있을 수 있는 단 하나의 오판 포인트</p>

<h2>📚 함께 보면 좋은 글</h2>
<ul>
  관련 글이 있으면 관련 과거 글 목록을 활용해 자연스러운 앵커 링크 삽입
  관련 과거 글이 없으면 {fallback_related_html} 사용
</ul>

<p style="font-size: 0.9em; color: gray; margin-top: 30px;">원본 출처: <a href="{html.escape(article.url)}" target="_blank" rel="noopener noreferrer">{html.escape(article.source_name)}</a></p>

<br>

<p style="font-size: 0.85em; color: #95a5a6; text-align: center;">※ 본 분석은 의사결정 참고용 데이터이며, 최종 투자 책임은 사용자 본인에게 있습니다.</p>

기사 데이터:
- 제목: {article.title}
- 출처: {article.source_name}
- URL: {article.url}
- 감지된 티커 후보: {tickers}
- RSS 요약: {article.rss_summary[:700]}
- 제한적 원문 여부: {limited_source}
- 본문 품질 점수: {quality_score}
- 기사 유형 코드: {article_type}
- 기사 유형명: {article_type_name}
- 전처리된 본문: {article.clean_text}

관련 과거 글:
{related_text}
""".strip()


def keyword_score(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in lowered)


def classify_article_type(article: NewsArticle, tickers: list[str]) -> str:
    text = f"{article.title} {article.rss_summary} {article.clean_text} {' '.join(tickers)}".lower()
    strong_cashflow_keywords = [
        keyword
        for keyword in CASHFLOW_VALUE_KEYWORDS + CASHFLOW_VALUE_KEYWORDS_KO
        if keyword.lower() not in {"dividend", "배당"}
    ]
    cash_score = keyword_score(text, strong_cashflow_keywords)
    etf_score = keyword_score(text, ETF_DIVIDEND_KEYWORDS)
    ai_score = keyword_score(text, AI_SEMICONDUCTOR_KEYWORDS)

    if etf_score >= 2 or (etf_score >= 1 and "etf" in text):
        return "etf_dividend"
    if cash_score and ai_score:
        return "ai_semiconductor" if ai_score >= cash_score + 2 else "cashflow_value"
    if cash_score:
        return "cashflow_value"
    if ai_score:
        return "ai_semiconductor"
    if re.search(r"big tech|cloud|advertising|platform|regulation|capex|apple|microsoft|google|meta|amazon", text):
        return "bigtech"
    if re.search(r"consumer|retail|inventory|brand|pricing power|discretionary|staples|sales slowdown", text):
        return "consumer_cyclical"
    return "general_market"


def article_type_guide(article_type: str) -> str:
    guides = {
        "cashflow_value": "- 자동 추정 기사 유형: 현금흐름/가치주 기사\n- 자유현금흐름(FCF)\n- FCF Margin\n- ROIC\n- 자본배분\n- 자사주 매입\n- 배당\n- M&A\n- 현금흐름은 좋지만 성장성이 부족한 기업의 함정\n- \"현금을 많이 버는 기업\"과 \"좋은 투자 대상\"의 차이\n- 현금 창출력과 성장 지속성 사이의 균형",
        "ai_semiconductor": "- 자동 추정 기사 유형: AI/반도체 기사\n- AI 학습 vs AI 추론\n- GPU, ASIC, 메모리, HBM, 저가 메모리\n- 엔비디아, AMD, 인텔, 빅테크 자체 칩\n- 공급망, 전력, 데이터센터 CAPEX\n- 성능보다 TCO가 중요한 구간\n- 반도체 사이클과 고객 집중 리스크",
        "etf_dividend": "- 자동 추정 기사 유형: ETF/배당 기사\n- 총수익률\n- 배당 착시\n- NAV 훼손\n- 비용률\n- 기초자산 리스크\n- 분배금 지속 가능성",
        "consumer_cyclical": "- 자동 추정 기사 유형: 소비재/경기민감주 기사\n- 소비 사이클\n- 재고 부담\n- 금리 민감도\n- 수요 둔화\n- 브랜드 파워\n- 가격 전가력",
        "bigtech": "- 자동 추정 기사 유형: 빅테크 기사\n- 클라우드\n- 광고\n- AI CAPEX\n- 플랫폼 락인\n- 규제 리스크\n- 주주환원",
    }
    return guides.get(article_type, "- 자동 추정 기사 유형: 일반 시장 뉴스\n- 기사에 확인된 사건\n- 산업 구조\n- 경쟁 구도\n- 수요와 비용 변수\n- 투자자가 확인해야 할 리스크")


def build_manual_gpt_prompt(article: NewsArticle, history: dict, tickers: list[str]) -> str:
    related_posts = find_related_posts(history, {"category": "미국증시"}, tickers)
    related_text = "\n".join(
        f"- {post['title']}: {post['url']}" for post in related_posts
    ) or "- 관련 과거 글 없음"
    limited_source = is_limited_source(article)
    quality_score = content_quality_score(article)
    article_type = classify_article_type(article, tickers)
    article_type_name = article_type_label(article_type)
    guide = article_type_guide(article_type)
    limited_source_instruction = (
        """- 제한적 원문 여부: True
- 원문이 Premium/부분 공개 기사이거나 본문 데이터가 부족합니다.
- 얇은 기사 요약문으로 작성하지 마세요.
- 제목과 공개 요약을 "시드 이슈"로만 사용하세요.
- 확인되지 않은 실적 수치, 목표주가, PER/PBR, EPS, 컨센서스, 애널리스트 의견, 성능 데이터는 절대 만들지 마세요.
- 일반 기업 소개로 분량을 채우지 말고, 구조적 원가 압박, 거시경제적 밸류에이션 변수, 경쟁 구도, 기관 투자자의 리스크 점검 프레임워크 중심으로 확장하세요.
- 필요한 경우 "기사 데이터만으로는 확인이 제한됩니다"라고 명시하세요."""
        if limited_source
        else
        """- 제한적 원문 여부: False
- 원문 본문이 충분하더라도 기사에 없는 재무 수치, 목표주가, 컨센서스, 공시 내용은 만들지 마세요.
- 단순 요약이 아니라 투자자 관점의 시장 기대치와 mispricing 가능성을 분석하세요."""
    )
    fallback_related_html = (
        "<li>관련 과거 글은 아직 없습니다. 향후 반도체, AI, 자유현금흐름, "
        "배당 ETF 관련 분석 글을 추가할 예정입니다.</li>"
    )

    return f"""
당신은 기관투자자(Buy-side) 스타일의 퀀트 기반 재무분석가이자 10년 차 자산운용사 매니저입니다.

목표:
- 아래 기사 데이터를 바탕으로 구글 검색 상위 노출을 노리는 한국어 Blogger/Tistory용 HTML 글을 작성하세요.
- 단순 요약이 아니라 시장 기대치 대비 mispricing이 발생한 투자 기회와 위험을 분석하세요.
- 수집되지 않은 재무 수치, 컨센서스, 목표주가, 애널리스트 의견, 공시 내용은 절대 지어내지 마세요.
- 자료에 없는 내용은 "기사 데이터만으로는 확인이 제한됩니다"라고 표현하세요.
- 데이터 공백이 있으면 "현재 공개된 RSS/본문 기준으로는 해당 수치 확인이 제한되며, 향후 공식 공시나 실적 자료 확인이 필요합니다"처럼 전문적으로 밝혀 주세요.
- 인사말, 감탄사, "첫째로", "요약하자면" 같은 진부한 표현 없이 바로 분석으로 들어가세요.

제한적 원문 처리:
{limited_source_instruction}
- 글 길이는 공백 제외 최소 2,000자 이상, 가능하면 2,500~3,500자 수준으로 작성하세요.
- 정보 이득이 낮은 일반론을 피하고, 독자가 투자 판단 전에 점검할 수 있는 분석 프레임워크를 제시하세요.

SEO 제목 작성 규칙:
- 검색자가 실제로 입력할 만한 키워드를 포함하세요.
- 단순 기사 제목 번역을 피하세요.
- 산업 구조, 투자 포인트, 경쟁 구도, 리스크를 반영하세요.
- 예:
  - "인텔 AI 반도체 재도전, 엔비디아 독주를 흔들 수 있을까?"
  - "현금흐름이 좋은 기업이 반드시 좋은 투자일까? FCF와 자본배분의 함정"
  - "AI 추론 시장이 커질수록 주목해야 할 반도체 투자 포인트"

기사 유형별 확장 가이드:
{guide}

내부 링크 지침:
- 관련 과거 글이 있으면 하단 목록에만 몰아넣지 말고, 본문 중간 문맥에 자연스러운 앵커 텍스트 링크로 최소 1개 이상 삽입하세요.
- 관련성이 낮은 글은 억지로 넣지 마세요.
- 관련 과거 글이 없으면 "함께 보면 좋은 글" 섹션에 아래 대체 문구를 사용하세요: {fallback_related_html}

FAQ 스니펫 지침:
- FAQ 질문은 <h3> 태그로 작성하세요.
- 답변은 바로 다음 <p>에 배치하고, 2~3문장 이내로 결론부터 명확히 쓰세요.
- 답변 첫 문장은 "정답 요약:"으로 시작하세요.

출력:
- Blogger/Tistory에 바로 붙여넣을 수 있는 HTML만 출력하세요.
- Markdown 설명, 코드블록, 별도 해설은 출력하지 마세요.

HTML 구조:
<h1>SEO 제목</h1>

<p>이 이슈를 왜 지금 봐야 하는지 시장 기대치와 엮은 강한 도입부 2~3문장</p>

<hr>

<blockquote style="background: #f9f9f9; border-left: 8px solid #007bff; padding: 15px; margin: 20px 0;">
  📌 <strong>기관 투자자 관점 핵심 3줄 요약</strong><br>
  • 핵심 포인트 1<br>
  • 핵심 포인트 2<br>
  • 핵심 포인트 3
</blockquote>

<h2>📊 시장 기대치와의 괴리 (Expectation Gap)</h2>
<p>시장 기대치와 실제 뉴스 사이의 차이를 Fact 중심으로 분석</p>

<h2>🏢 기업의 현재 위치와 경쟁 구도</h2>
<p>분석 대상 기업이 현재 산업 안에서 어떤 위치에 있는지, 관련 경쟁 구도와 함께 설명</p>

<h2>🧠 투자자가 봐야 할 구조적 변수</h2>
<p>기사 유형에 맞춰 원가 구조, 현금흐름, 자본배분, 공급망, 수요 사이클, 밸류에이션 민감도 등을 분석</p>

<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<h3>개인 투자자가 검색할 만한 핵심 질문</h3>
<p><strong>정답 요약:</strong> 구글 추천 스니펫에 적합하도록 2~3문장으로 결론부터 답변</p>

<h2>⚖️ 찬반 논리 점검 (Bull vs Bear Thesis)</h2>
<ul>
  <li><strong>상방 모멘텀 (Bull):</strong> 성장 모멘텀 또는 catalyst 중심 긍정 논리</li>
  <li><strong>하방 리스크 (Bear):</strong> 밸류에이션, 희석, 규제, 수요 둔화 등 반대 논리</li>
</ul>

<h2>💡 월가의 오판 포인트 (Market Mispricing)</h2>
<p style="color: #2c3e50; font-weight: bold; background: #f0f7ff; padding: 12px; border-radius: 5px;">시장이 놓치고 있을 수 있는 단 하나의 오판 포인트</p>

<h2>📚 함께 보면 좋은 글</h2>
<ul>
  관련 과거 글이 있으면 자연스러운 앵커 링크 삽입
  관련 과거 글이 없으면 {fallback_related_html} 사용
</ul>

<p style="font-size: 0.9em; color: gray; margin-top: 30px;">원본 출처: <a href="{html.escape(article.url)}" target="_blank" rel="noopener noreferrer nofollow">{html.escape(article.source_name)}</a></p>

<br>

<p style="font-size: 0.85em; color: #95a5a6; text-align: center;">※ 본 분석은 의사결정 참고용 데이터이며, 최종 투자 책임은 사용자 본인에게 있습니다.</p>

기사 데이터:
- 제목: {article.title}
- 출처: {article.source_name}
- URL: {article.url}
- 감지된 티커 후보: {tickers}
- RSS 요약: {article.rss_summary[:700]}
- 제한적 원문 여부: {limited_source}
- 본문 품질 점수: {quality_score}
- 기사 유형 코드: {article_type}
- 기사 유형명: {article_type_name}
- 전처리된 본문: {article.clean_text}

관련 과거 글:
{related_text}
""".strip()


def save_manual_prompt_output(article: NewsArticle, history: dict, tickers: list[str]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^A-Za-z0-9가-힣_-]+", "_", article.title)[:60]
    path = OUTPUT_DIR / f"{timestamp}_manual_prompt_{safe_title}.md"
    path.write_text(build_manual_gpt_prompt(article, history, tickers), encoding="utf-8")
    logging.info("Manual GPT prompt saved: %s", path)


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
    metrics.success = (
        metrics.uploaded_posts > 0
        or metrics.dry_run_outputs > 0
        or metrics.manual_prompt_outputs > 0
        or metrics.failed_articles == 0
    )
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

    if is_thin_article(article) or is_limited_source(article):
        if os.getenv("MANUAL_PROMPT_MODE", "false").lower() == "true" and article.rss_summary.strip():
            logging.info("Limited/thin article accepted for manual prompt mode: %s", article.title)
        else:
            metrics.skipped_thin_articles += 1
            logging.info("Skipping thin article before OpenAI call: %s", article.title)
            return False

    seen_cache.add(article_cache_key(article))
    tickers = detect_tickers(article)

    if os.getenv("MANUAL_PROMPT_MODE", "false").lower() == "true":
        save_manual_prompt_output(article, history, tickers)
        metrics.manual_prompt_outputs += 1
        return True

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
            if metrics.uploaded_posts + metrics.dry_run_outputs + metrics.manual_prompt_outputs >= max_posts:
                break
            if metrics.quota_or_rate_limit_blocked:
                logging.warning("Stopping early because OpenAI quota or rate limit is blocked.")
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

    should_require_post = os.getenv("REQUIRE_POST_SUCCESS", "true").lower() == "true"
    is_real_upload_mode = os.getenv("DRY_RUN", "false").lower() != "true"
    is_manual_prompt_mode = os.getenv("MANUAL_PROMPT_MODE", "false").lower() == "true"
    if should_require_post and is_real_upload_mode and not is_manual_prompt_mode and metrics.uploaded_posts == 0:
        logging.error(
            "No Blogger post was created. Marking workflow as failed to avoid a false success signal."
        )
        raise SystemExit(1)


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
