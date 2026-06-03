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
AI_MANUFACTURING_KEYWORDS = [
    "tsmc",
    "lithography",
    "process simulation",
    "fab operations",
    "defect inspection",
    "yield",
    "yield rate",
    "digital twin",
    "wafer",
    "aoi",
    "automated optical inspection",
    "manufacturing ai",
    "factory ai",
    "process optimization",
    "semiconductor manufacturing",
    "manufacturing capex",
]
AI_ENTERPRISE_KEYWORDS = [
    "enterprise ai",
    "ai agent",
    "workflow automation",
    "productivity software",
    "enterprise customer",
    "saas",
    "cloud workload",
    "m365",
    "teams",
    "azure",
]
AI_HEALTHCARE_KEYWORDS = [
    "healthcare ai",
    "medical ai",
    "medical center",
    "medical centers",
    "clinician",
    "clinicians",
    "hospital",
    "patient",
    "diagnostic",
    "foxconn",
    "ai agent systems",
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
    labels = {
        "ai_manufacturing": "AI 제조혁신 기사",
        "ai_enterprise": "AI 엔터프라이즈 기사",
        "ai_healthcare": "AI 헬스케어 기사",
    }
    return labels.get(article_type) or ARTICLE_TYPE_LABELS.get(article_type, ARTICLE_TYPE_LABELS["general_market"])


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


def classify_article_topics(article: NewsArticle, tickers: list[str]) -> tuple[str, str | None]:
    text = f"{article.title} {article.rss_summary} {article.clean_text} {' '.join(tickers)}".lower()
    strong_cashflow_keywords = [
        keyword
        for keyword in CASHFLOW_VALUE_KEYWORDS + CASHFLOW_VALUE_KEYWORDS_KO
        if keyword.lower() not in {"dividend", "배당"}
    ]
    cash_score = keyword_score(text, strong_cashflow_keywords)
    etf_score = keyword_score(text, ETF_DIVIDEND_KEYWORDS)
    ai_score = keyword_score(text, AI_SEMICONDUCTOR_KEYWORDS)
    manufacturing_score = keyword_score(text, AI_MANUFACTURING_KEYWORDS)
    enterprise_score = keyword_score(text, AI_ENTERPRISE_KEYWORDS)
    healthcare_score = keyword_score(text, AI_HEALTHCARE_KEYWORDS)

    if etf_score >= 2 or (etf_score >= 1 and "etf" in text):
        primary = "etf_dividend"
    elif manufacturing_score >= 2:
        primary = "ai_manufacturing"
    elif enterprise_score >= 2:
        primary = "ai_enterprise"
    elif healthcare_score >= 2:
        primary = "ai_healthcare"
    elif cash_score and ai_score:
        primary = "ai_semiconductor" if ai_score >= cash_score + 2 else "cashflow_value"
    elif cash_score:
        primary = "cashflow_value"
    elif ai_score:
        primary = "ai_semiconductor"
    elif re.search(r"big tech|cloud|advertising|platform|regulation|capex|apple|microsoft|google|meta|amazon", text):
        primary = "bigtech"
    elif re.search(r"consumer|retail|inventory|brand|pricing power|discretionary|staples|sales slowdown", text):
        primary = "consumer_cyclical"
    else:
        primary = "general_market"

    secondary_candidates = [
        ("ai_manufacturing", manufacturing_score),
        ("ai_healthcare", healthcare_score),
        ("ai_enterprise", enterprise_score),
        ("ai_semiconductor", ai_score),
        ("cashflow_value", cash_score),
        ("etf_dividend", etf_score),
    ]
    secondary = next((topic for topic, score in secondary_candidates if topic != primary and score >= 2), None)
    return primary, secondary


def classify_article_type(article: NewsArticle, tickers: list[str]) -> str:
    primary, _ = classify_article_topics(article, tickers)
    return primary


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
        "cashflow_value": "- 자동 추정 기사 유형: 현금흐름/가치주 기사\n- 자유현금흐름(FCF), FCF Margin, ROIC\n- 자본배분, 자사주 매입, 배당, M&A\n- 현금흐름은 좋지만 성장성이 부족한 기업의 함정\n- \"현금을 많이 버는 기업\"과 \"좋은 투자 대상\"의 차이\n- 현금 창출력과 성장 지속성 사이의 균형",
        "ai_semiconductor": "- 자동 추정 기사 유형: AI/반도체 기사\n- AI 학습 vs AI 추론\n- GPU, ASIC, 메모리, HBM, 저가 메모리\n- 엔비디아, AMD, 인텔, 빅테크 자체 칩\n- 공급망, 전력, 데이터센터 CAPEX\n- 성능보다 TCO가 중요한 구간\n- 반도체 사이클과 고객 집중 리스크\n- Yield Rate Improvements(수율 개선)\n- Lithography Computing Overhead(리소그래피 연산 부하)\n- Wafer Margin Optimization(웨이퍼당 마진 최적화)\n- OPEX Reduction via AOI 자동화\n- Depreciation Costs of Advanced Fabs(첨단 팹 감가상각비 부담)",
        "etf_dividend": "- 자동 추정 기사 유형: ETF/배당 기사\n- 총수익률, 배당 착시, NAV 훼손\n- 비용률, 기초자산 리스크\n- 분배금 지속 가능성",
        "consumer_cyclical": "- 자동 추정 기사 유형: 소비재/경기민감주 기사\n- 소비 사이클, 재고 부담, 금리 민감도\n- 수요 둔화, 브랜드 파워, 가격 전가력",
        "bigtech": "- 자동 추정 기사 유형: 빅테크 기사\n- 클라우드, 광고, AI CAPEX\n- 플랫폼 락인, 규제 리스크, 주주환원\n- 기술 부채와 보안 리스크가 기업 고객 유지율에 미치는 영향\n- Cloud Workload Unit Economics\n- Security Debt와 Churn Risk\n- AI CAPEX의 감가상각 부담\n- ARPU 방어력과 엔터프라이즈 락인",
    }
    return guides.get(article_type, "- 자동 추정 기사 유형: 일반 시장 뉴스\n- 기사에 확인된 사건\n- 산업 구조\n- 경쟁 구도\n- 수요와 비용 변수\n- 투자자가 확인해야 할 리스크")


def fallback_related_links_html(article_type: str) -> str:
    category_links = {
        "cashflow_value": [
            ("/category/재무분석", "💡 미국 기업 자유현금흐름(FCF) 및 자본배분 분석 모음 보기"),
            ("/category/미국주식", "📈 빅테크와 가치주의 주주환원 전략 분석 보기"),
        ],
        "ai_semiconductor": [
            ("/category/AI·반도체", "💡 글로벌 AI 및 반도체 공급망 밸류에이션 분석 시리즈 보기"),
            ("/category/미국주식", "📈 빅테크 AI CAPEX와 데이터센터 투자 사이클 보기"),
        ],
        "etf_dividend": [
            ("/category/ETF·배당", "💡 배당 ETF의 총수익률과 분배금 지속 가능성 분석 보기"),
            ("/category/미국주식", "📈 미국 인컴형 자산의 리스크 점검 글 모음 보기"),
        ],
    }
    links = category_links.get(
        article_type,
        [
            ("/category/미국주식", "📈 미국 주식 시장 이슈 분석 모음 보기"),
            ("/category/재무분석", "💡 기업 실적과 현금흐름 분석 글 모음 보기"),
        ],
    )
    return "\n".join(f'  <li><a href="{url}">{label}</a></li>' for url, label in links)


def build_manual_gpt_prompt(article: NewsArticle, history: dict, tickers: list[str]) -> str:
    related_posts = find_related_posts(history, {"category": "미국증시"}, tickers)
    related_text = "\n".join(
        f"- {post['title']}: {post['url']}" for post in related_posts
    ) or "- 제공된 과거 글 없음. 아래 카테고리 링크 대체 규칙을 사용하세요."
    limited_source = is_limited_source(article)
    quality_score = content_quality_score(article)
    article_type = classify_article_type(article, tickers)
    article_type_name = article_type_label(article_type)
    guide = article_type_guide(article_type)
    fallback_links = fallback_related_links_html(article_type)
    limited_source_instruction = (
        """- 제한적 원문 여부: True
- 원문이 Premium/부분 공개 기사이거나 본문 데이터가 부족합니다.
- 얇은 기사 요약문으로 작성하지 마세요. 제목과 공개 요약은 "시드 이슈"로만 사용하세요.
- 본문에서 "기사 데이터만으로는 확인이 제한됩니다" 같은 방어 문구를 반복하지 마세요.
- 데이터 한계는 마지막 투자 책임 고지 문단에서 1회만 언급하세요.
- 확인되지 않은 실적 수치, 목표주가, PER/PBR, EPS, 컨센서스, 애널리스트 의견, 성능 데이터는 절대 만들지 마세요.
- 일반 기업 소개로 분량을 채우지 말고, 구조적 원가 압박, 거시경제적 밸류에이션 변수, 경쟁 구도, 기술 부채, 기관 투자자의 리스크 점검 프레임워크 중심으로 확장하세요."""
        if limited_source
        else
        """- 제한적 원문 여부: False
- 원문 본문이 충분하더라도 기사에 없는 재무 수치, 목표주가, 컨센서스, 공시 내용은 만들지 마세요.
- 단순 요약이 아니라 투자자 관점의 시장 기대치와 mispricing 가능성을 분석하세요.
- 데이터 한계 문구는 필요할 때 마지막 투자 책임 고지 문단에서만 1회 사용하세요."""
    )

    return f"""
당신은 기관투자자(Buy-side) 스타일의 퀀트 기반 재무분석가이자 10년 차 자산운용사 매니저입니다.

목표:
- 아래 기사 데이터를 바탕으로 구글 검색 상위 노출을 노리는 한국어 Blogger/Tistory용 HTML 글을 작성하세요.
- 단순 요약이 아니라 시장 기대치 대비 mispricing이 발생한 투자 기회와 위험을 분석하세요.
- 수집되지 않은 재무 수치, 컨센서스, 목표주가, 애널리스트 의견, 공시 내용은 절대 지어내지 마세요.
- 본문에서는 방어적인 공백 선언을 반복하지 말고, 데이터 공백이 있으면 기관투자자 분석 프레임워크로 자연스럽게 전환하세요.
- 인사말, 감탄사, "첫째로", "요약하자면" 같은 진부한 표현 없이 바로 분석으로 들어가세요.

제한적 원문 처리:
{limited_source_instruction}
- 글 길이는 공백 제외 최소 2,000자 이상, 가능하면 2,500~3,500자 수준으로 작성하세요.
- 정보 이득이 낮은 일반론을 피하고, 독자가 투자 판단 전에 점검할 수 있는 구조적 변수와 리스크 프레임워크를 제시하세요.

가독성 및 리스트 작성 규칙:
- <li> 항목은 평문으로 시작하지 말고 반드시 <strong>핵심 키워드</strong>: 형식으로 시작하세요.
- 경쟁 구도, 구조적 변수, 투자 체크포인트처럼 순서와 인과관계가 중요한 영역은 <ul>보다 <ol>을 우선 사용하세요.
- 모바일 독자가 빠르게 훑을 수 있도록 각 항목은 핵심 명사와 해석을 분리하세요.
- 예: <li><strong>엔비디아(NVIDIA)</strong>: GPU와 CUDA 생태계를 넘어 파운드리 공정 생산성까지 확장하는 AI 플랫폼 전략</li>

전문 리서치 문체 규칙:
- 나쁜 표현: "현재 기사 데이터만으로는 수치 확인이 제한되므로 향후 실적을 봐야 합니다."
- 좋은 표현: "단기적으로 이 이슈가 가시적인 밸류에이션 리레이팅으로 이어지기 위해서는, 향후 분기 실적(Form 10-Q)에서 고객 이탈률, ARPU, 마진 방어 여부를 추적 관찰해야 합니다."
- 위 좋은 표현의 톤을 따르되, 기사에 없는 수치를 만들지 마세요.

SEO 제목 작성 규칙:
- 검색자가 실제로 입력할 만한 키워드를 포함하세요.
- 단순 기사 제목 번역을 피하세요.
- 산업 구조, 투자 포인트, 경쟁 구도, 리스크를 반영하세요.

기사 유형별 확장 가이드:
{guide}

내부 링크 지침:
- 관련 과거 글이 있으면 하단 목록에만 몰아넣지 말고, 본문 중간 문맥에 자연스러운 앵커 텍스트 링크로 최소 1개 이상 삽입하세요.
- 관련성이 낮은 글은 억지로 넣지 마세요.
- 제공된 과거 글이 없으면 부재 안내 문장을 쓰지 말고 아래 카테고리 링크를 사용하세요.

카테고리 대체 링크:
<h2>📚 함께 보면 좋은 글</h2>
<ul>
{fallback_links}
</ul>

FAQ 스니펫 지침:
- FAQ 질문은 <h3> 태그로 작성하세요.
- 첫 번째 답변 문단은 바로 다음 <p><strong>정답 요약:</strong> ...</p>에 배치하세요.
- 정답 요약 문단은 정확히 1~2문장, 가능하면 30단어 이하로 직접적인 결론만 담으세요.
- 보충 논리, 배경, 추적 지표는 반드시 바로 아래 별도 <p> 문단으로 분리하세요.
- "아직 모릅니다" 식의 회피형 답변은 피하고, 확인 가능한 범위에서 명확한 결론과 전략적 해석을 제시하세요.

출력:
- Blogger/Tistory에 바로 붙여넣을 수 있는 HTML만 출력하세요.
- Markdown 설명, 코드블록, 별도 해설은 출력하지 마세요.
- 아래 HTML 구조의 모든 섹션을 생략하지 마세요.

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

<h2>⚙️ 기술 전략과 원가 구조의 의미 (Technical Strategy & Cost Structure)</h2>
<p>수치가 부족해도 섹션을 삭제하지 말고, 기사 유형에 맞춰 클라우드 인프라, 기술 부채, 보안 리스크, 공급망, 현금흐름, 자본배분, 비용 구조, 고객 이탈률 같은 구조적 변수를 분석</p>
<ol>
  <li><strong>구조적 비용 변수</strong>: 기사 유형에 맞는 비용 또는 효율성 레버를 설명</li>
  <li><strong>마진 민감도</strong>: 수율, 자동화, 기술 부채, 고객 이탈률, CAPEX 감가상각 등 마진에 영향을 주는 변수를 설명</li>
  <li><strong>투자자 체크포인트</strong>: 향후 실적 또는 공시에서 추적할 지표를 설명</li>
</ol>

<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<h3>개인 투자자가 검색할 만한 핵심 질문</h3>
<p><strong>정답 요약:</strong> 구글 추천 스니펫에 적합하도록 1~2문장으로 결론만 답변</p>
<p>보충 논리와 추적해야 할 지표를 별도 문단으로 설명</p>

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
  제공된 과거 글이 없으면 위 카테고리 대체 링크 사용
</ul>

<div style="font-size: 0.9em; color: #6b7280; margin-top: 30px; text-align: right;">
  원본 출처: <a href="{html.escape(article.url)}" target="_blank" rel="noopener noreferrer nofollow">{html.escape(article.source_name)}</a>
</div>

<br>

<div style="background: #f8fafc; color: #64748b; font-size: 0.8em; line-height: 1.5; text-align: center; padding: 12px; border-radius: 6px; margin-top: 12px;">
  ※ 본 분석은 의사결정 참고용 데이터입니다. 일부 원문 데이터가 제한적인 경우 최종 투자 판단 전 공식 공시와 실적 자료 확인이 필요하며, 최종 투자 책임은 사용자 본인에게 있습니다.
</div>

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


def classify_article_topics(article: NewsArticle, tickers: list[str]) -> tuple[str, str | None]:
    text = f"{article.title} {article.rss_summary} {article.clean_text} {' '.join(tickers)}".lower()
    strong_cashflow_keywords = [
        keyword
        for keyword in CASHFLOW_VALUE_KEYWORDS + CASHFLOW_VALUE_KEYWORDS_KO
        if keyword.lower() not in {"dividend", "배당"}
    ]
    cash_score = keyword_score(text, strong_cashflow_keywords)
    etf_score = keyword_score(text, ETF_DIVIDEND_KEYWORDS)
    ai_score = keyword_score(text, AI_SEMICONDUCTOR_KEYWORDS)
    manufacturing_score = keyword_score(text, AI_MANUFACTURING_KEYWORDS)
    enterprise_score = keyword_score(text, AI_ENTERPRISE_KEYWORDS)
    healthcare_score = keyword_score(text, AI_HEALTHCARE_KEYWORDS)

    if etf_score >= 2 or (etf_score >= 1 and "etf" in text):
        primary = "etf_dividend"
    elif manufacturing_score >= 2:
        primary = "ai_manufacturing"
    elif enterprise_score >= 2:
        primary = "ai_enterprise"
    elif healthcare_score >= 2:
        primary = "ai_healthcare"
    elif cash_score and ai_score:
        primary = "ai_semiconductor" if ai_score >= cash_score + 2 else "cashflow_value"
    elif cash_score:
        primary = "cashflow_value"
    elif ai_score:
        primary = "ai_semiconductor"
    elif re.search(r"big tech|cloud|advertising|platform|regulation|capex|apple|microsoft|google|meta|amazon", text):
        primary = "bigtech"
    elif re.search(r"consumer|retail|inventory|brand|pricing power|discretionary|staples|sales slowdown", text):
        primary = "consumer_cyclical"
    else:
        primary = "general_market"

    secondary_candidates = [
        ("ai_manufacturing", manufacturing_score),
        ("ai_healthcare", healthcare_score),
        ("ai_enterprise", enterprise_score),
        ("ai_semiconductor", ai_score),
        ("cashflow_value", cash_score),
        ("etf_dividend", etf_score),
    ]
    secondary = next((topic for topic, score in secondary_candidates if topic != primary and score >= 2), None)
    return primary, secondary


def classify_article_type(article: NewsArticle, tickers: list[str]) -> str:
    primary, _ = classify_article_topics(article, tickers)
    return primary


def article_type_guide(article_type: str) -> str:
    guides = {
        "cashflow_value": "- 자동 추정 기사 유형: 현금흐름/가치주 기사\n- 자유현금흐름(FCF), FCF Margin, ROIC\n- 자본배분, 자사주 매입, 배당, M&A\n- 현금흐름은 좋지만 성장성이 부족한 기업의 함정\n- \"현금을 많이 버는 기업\"과 \"좋은 투자 대상\"의 차이\n- 현금 창출력과 성장 지속성 사이의 균형",
        "ai_semiconductor": "- 자동 추정 기사 유형: AI/반도체 기사\n- AI 학습 vs AI 추론\n- GPU, ASIC, 메모리, HBM, 저가 메모리\n- 엔비디아, AMD, 인텔, 빅테크 자체 칩\n- 공급망, 전력, 데이터센터 CAPEX\n- 성능보다 TCO가 중요한 구간\n- 반도체 사이클과 고객 집중 리스크",
        "ai_manufacturing": "- 자동 추정 기사 유형: AI 제조혁신 기사\n- AI가 반도체 제조공정에 미치는 영향\n- Lithography 최적화\n- Yield(수율) 개선\n- 결함 검사 자동화\n- Fab 운영 효율화\n- 디지털 트윈\n- 공정 시뮬레이션\n- 제조 CAPEX 절감\n- 생산성 향상\n- AI 소프트웨어의 제조업 확산\n- 데이터센터 외 TAM 확대\n- AI의 산업 침투율\n- 생산성 기반 투자수익률\n- 제조업 디지털 전환",
        "ai_enterprise": "- 자동 추정 기사 유형: AI 엔터프라이즈 기사\n- AI Agent와 업무 자동화\n- SaaS/클라우드 워크로드 확장\n- 기업 고객 락인과 ARPU 방어\n- 보안 리스크와 기술 부채\n- Churn Risk와 도입 비용\n- 엔터프라이즈 생산성 투자수익률",
        "ai_healthcare": "- 자동 추정 기사 유형: AI 헬스케어 기사\n- 의료기관 AI Agent 도입\n- 임상의 업무 보조와 워크플로우 자동화\n- 환자 데이터 보안과 규제 리스크\n- 병원 운영 효율화\n- 진단 보조와 의료 생산성\n- 헬스케어 AI의 상용화 장벽",
        "etf_dividend": "- 자동 추정 기사 유형: ETF/배당 기사\n- 총수익률, 배당 착시, NAV 훼손\n- 비용률, 기초자산 리스크\n- 분배금 지속 가능성",
        "consumer_cyclical": "- 자동 추정 기사 유형: 소비재/경기민감주 기사\n- 소비 사이클, 재고 부담, 금리 민감도\n- 수요 둔화, 브랜드 파워, 가격 전가력",
        "bigtech": "- 자동 추정 기사 유형: 빅테크 기사\n- 클라우드, 광고, AI CAPEX\n- 플랫폼 락인, 규제 리스크, 주주환원\n- 기술 부채와 보안 리스크가 기업 고객 유지율에 미치는 영향",
    }
    return guides.get(article_type, "- 자동 추정 기사 유형: 일반 시장 뉴스\n- 기사에 확인된 사건\n- 산업 구조\n- 경쟁 구도\n- 수요와 비용 변수\n- 투자자가 확인해야 할 리스크")


def fallback_related_links_html(article_type: str) -> str:
    category_links = {
        "cashflow_value": [
            ("/category/재무분석", "💡 미국 기업 자유현금흐름(FCF) 및 자본배분 분석 모음 보기"),
            ("/category/미국주식", "📈 빅테크와 가치주의 주주환원 전략 분석 보기"),
        ],
        "ai_semiconductor": [
            ("/category/AI·반도체", "💡 글로벌 AI 및 반도체 공급망 밸류에이션 분석 시리즈 보기"),
            ("/category/미국주식", "📈 빅테크 AI CAPEX와 데이터센터 투자 사이클 보기"),
        ],
        "ai_manufacturing": [
            ("/category/AI·제조혁신", "🏭 AI 제조혁신과 반도체 공정 생산성 분석 보기"),
            ("/category/AI·반도체", "💡 반도체 수율 경쟁과 파운드리 밸류체인 분석 보기"),
        ],
        "ai_enterprise": [
            ("/category/AI·엔터프라이즈", "🏢 AI Agent와 기업 생산성 자동화 분석 보기"),
            ("/category/미국주식", "📈 빅테크 클라우드와 AI 소프트웨어 투자 포인트 보기"),
        ],
        "ai_healthcare": [
            ("/category/AI·헬스케어", "🏥 의료 AI Agent와 헬스케어 자동화 분석 보기"),
            ("/category/AI", "💡 산업별 AI 침투율과 생산성 투자 테마 보기"),
        ],
        "etf_dividend": [
            ("/category/ETF·배당", "💡 배당 ETF의 총수익률과 분배금 지속 가능성 분석 보기"),
            ("/category/미국주식", "📈 미국 인컴형 자산의 리스크 점검 글 모음 보기"),
        ],
    }
    links = category_links.get(
        article_type,
        [
            ("/category/미국주식", "📈 미국 주식 시장 이슈 분석 모음 보기"),
            ("/category/재무분석", "💡 기업 실적과 현금흐름 분석 글 모음 보기"),
        ],
    )
    return "\n".join(f'  <li><a href="{url}">{label}</a></li>' for url, label in links)


def build_manual_gpt_prompt(article: NewsArticle, history: dict, tickers: list[str]) -> str:
    related_posts = find_related_posts(history, {"category": "미국증시"}, tickers)
    related_text = "\n".join(
        f"- {post['title']}: {post['url']}" for post in related_posts
    ) or "- 제공된 과거 글 없음. 아래 카테고리 링크 대체 규칙을 사용하세요."
    limited_source = is_limited_source(article)
    quality_score = content_quality_score(article)
    primary_topic, secondary_topic = classify_article_topics(article, tickers)
    primary_topic_name = article_type_label(primary_topic)
    secondary_topic_name = article_type_label(secondary_topic) if secondary_topic else "없음"
    guide = article_type_guide(primary_topic)
    if secondary_topic:
        guide = f"{guide}\n\n보조 주제 확장 가이드:\n{article_type_guide(secondary_topic)}"
    fallback_links = fallback_related_links_html(primary_topic)
    limited_source_instruction = (
        """- 제한적 원문 여부: True
- 원문이 Premium/부분 공개 기사이거나 본문 데이터가 부족합니다.
- 얇은 기사 요약문으로 작성하지 마세요. 제목과 공개 요약은 "시드 이슈"로만 사용하세요.
- 본문에서 방어 문구를 반복하지 말고, 데이터 한계는 마지막 투자 책임 고지 문단에서 1회만 언급하세요.
- 전체 글 분량 비율은 기사 요약 10% 이하, 산업 구조 분석 40%, 투자 프레임워크 30%, Bull/Bear/Mispricing 20%로 작성하세요.
- 확인되지 않은 실적 수치, 목표주가, PER/PBR, EPS, 컨센서스, 애널리스트 의견, 성능 데이터는 절대 만들지 마세요.
- 일반 기업 소개로 분량을 채우지 말고, primary_topic과 secondary_topic에 맞는 산업 침투율, 생산성, 비용 구조, 규제 리스크, TAM 확장 프레임워크로 확장하세요."""
        if limited_source
        else
        """- 제한적 원문 여부: False
- 원문 본문이 충분하더라도 기사에 없는 재무 수치, 목표주가, 컨센서스, 공시 내용은 만들지 마세요.
- 단순 요약이 아니라 투자자 관점의 시장 기대치와 mispricing 가능성을 분석하세요.
- 데이터 한계 문구는 필요할 때 마지막 투자 책임 고지 문단에서만 1회 사용하세요."""
    )

    return f"""
당신은 기관투자자(Buy-side) 스타일의 퀀트 기반 재무분석가이자 10년 차 자산운용사 매니저입니다.

목표:
- 아래 기사 데이터를 바탕으로 구글 검색 상위 노출을 노리는 한국어 Blogger/Tistory용 HTML 글을 작성하세요.
- 단순 요약이 아니라 시장 기대치 대비 mispricing이 발생한 투자 기회와 위험을 분석하세요.
- 수집되지 않은 재무 수치, 컨센서스, 목표주가, 애널리스트 의견, 공시 내용은 절대 지어내지 마세요.
- AI 기사는 GPU/데이터센터 일반론으로 자동 확장하지 말고, primary_topic과 secondary_topic이 가리키는 실제 침투 영역을 중심으로 작성하세요.
- 인사말, 감탄사, "첫째로", "요약하자면" 같은 진부한 표현 없이 바로 분석으로 들어가세요.

제한적 원문 처리:
{limited_source_instruction}
- 글 길이는 공백 제외 최소 2,000자 이상, 가능하면 2,500~3,500자 수준으로 작성하세요.
- 정보 이득이 낮은 일반론을 피하고, 독자가 투자 판단 전에 점검할 수 있는 구조적 변수와 리스크 프레임워크를 제시하세요.

주제 제어 규칙:
- primary_topic이 ai_manufacturing이면 엔비디아/GPU/데이터센터 일반론보다 TSMC, lithography, process simulation, fab operations, defect inspection, yield, digital twin, manufacturing CAPEX를 우선하세요.
- secondary_topic이 ai_healthcare이면 Foxconn, medical centers, clinicians, AI agent systems, 의료 워크플로우 자동화, 규제 리스크를 별도 문단에서 짚으세요.
- primary_topic과 secondary_topic이 모두 있으면 본문 제목과 도입부에서 두 주제를 연결해 "AI가 제조공정과 산업현장으로 확산되는 사례"로 해석하세요.

SEO 제목 작성 규칙:
- 검색자가 실제로 입력할 만한 키워드를 포함하세요.
- 단순 기사 제목 번역을 피하세요.
- 예:
  - "TSMC가 엔비디아 AI를 공장에 도입한 이유, 반도체 수율 경쟁이 시작됐다"
  - "AI는 이제 반도체 공장을 운영한다: 엔비디아와 TSMC 협력의 진짜 의미"
  - "엔비디아 AI가 GPU 판매를 넘어 제조공정으로 확산되는 이유"

기사 유형별 확장 가이드:
{guide}

가독성 및 리스트 작성 규칙:
- <li> 항목은 평문으로 시작하지 말고 반드시 <strong>핵심 키워드</strong>: 형식으로 시작하세요.
- 경쟁 구도, 구조적 변수, 투자 체크포인트처럼 순서와 인과관계가 중요한 영역은 <ul>보다 <ol>을 우선 사용하세요.

내부 링크 지침:
- 관련 과거 글이 있으면 본문 중간 문맥에 자연스러운 앵커 텍스트 링크로 최소 1개 이상 삽입하세요.
- 제공된 과거 글이 없으면 부재 안내 문장을 쓰지 말고 아래 카테고리 링크를 사용하세요.

카테고리 대체 링크:
<h2>📚 함께 보면 좋은 글</h2>
<ul>
{fallback_links}
</ul>

FAQ 스니펫 지침:
- FAQ 질문은 <h3> 태그로 작성하세요.
- 첫 번째 답변 문단은 바로 다음 <p><strong>정답 요약:</strong> ...</p>에 배치하세요.
- 정답 요약 문단은 정확히 1~2문장, 가능하면 30단어 이하로 직접적인 결론만 담으세요.
- 보충 논리, 배경, 추적 지표는 반드시 바로 아래 별도 <p> 문단으로 분리하세요.

출력:
- Blogger/Tistory에 바로 붙여넣을 수 있는 HTML만 출력하세요.
- Markdown 설명, 코드블록, 별도 해설은 출력하지 마세요.
- 아래 HTML 구조의 모든 섹션을 생략하지 마세요.

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
<h2>⚙️ 기술 전략과 원가 구조의 의미 (Technical Strategy & Cost Structure)</h2>
<p>수치가 부족해도 섹션을 삭제하지 말고, primary_topic에 맞는 구조적 변수를 분석</p>
<ol>
  <li><strong>구조적 비용 변수</strong>: 기사 유형에 맞는 비용 또는 효율성 레버를 설명</li>
  <li><strong>마진 민감도</strong>: 수율, 자동화, 기술 부채, 고객 이탈률, CAPEX 감가상각 등 마진에 영향을 주는 변수를 설명</li>
  <li><strong>투자자 체크포인트</strong>: 향후 실적 또는 공시에서 추적할 지표를 설명</li>
</ol>
<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<h3>개인 투자자가 검색할 만한 핵심 질문</h3>
<p><strong>정답 요약:</strong> 구글 추천 스니펫에 적합하도록 1~2문장으로 결론만 답변</p>
<p>보충 논리와 추적해야 할 지표를 별도 문단으로 설명</p>
<h2>⚖️ 찬반 논리 점검 (Bull vs Bear Thesis)</h2>
<ul>
  <li><strong>상방 모멘텀 (Bull)</strong>: 성장 모멘텀 또는 catalyst 중심 긍정 논리</li>
  <li><strong>하방 리스크 (Bear)</strong>: 밸류에이션, 희석, 규제, 수요 둔화 등 반대 논리</li>
</ul>
<h2>💡 월가의 오판 포인트 (Market Mispricing)</h2>
<p style="color: #2c3e50; font-weight: bold; background: #f0f7ff; padding: 12px; border-radius: 5px;">시장이 놓치고 있을 수 있는 단 하나의 오판 포인트</p>
<h2>📚 함께 보면 좋은 글</h2>
<ul>
  관련 과거 글이 있으면 자연스러운 앵커 링크 삽입
  제공된 과거 글이 없으면 위 카테고리 대체 링크 사용
</ul>
<div style="font-size: 0.9em; color: #6b7280; margin-top: 30px; text-align: right;">
  원본 출처: <a href="{html.escape(article.url)}" target="_blank" rel="noopener noreferrer nofollow">{html.escape(article.source_name)}</a>
</div>
<br>
<div style="background: #f8fafc; color: #64748b; font-size: 0.8em; line-height: 1.5; text-align: center; padding: 12px; border-radius: 6px; margin-top: 12px;">
  ※ 본 분석은 의사결정 참고용 데이터입니다. 일부 원문 데이터가 제한적인 경우 최종 투자 판단 전 공식 공시와 실적 자료 확인이 필요하며, 최종 투자 책임은 사용자 본인에게 있습니다.
</div>

기사 데이터:
- 제목: {article.title}
- 출처: {article.source_name}
- URL: {article.url}
- 감지된 티커 후보: {tickers}
- RSS 요약: {article.rss_summary[:700]}
- 제한적 원문 여부: {limited_source}
- 본문 품질 점수: {quality_score}
- primary_topic: {primary_topic}
- primary_topic_name: {primary_topic_name}
- secondary_topic: {secondary_topic or "없음"}
- secondary_topic_name: {secondary_topic_name}
- 전처리된 본문: {article.clean_text}

관련 과거 글:
{related_text}
""".strip()


def investor_concerns_for_topic(article_type: str, secondary_topic: str | None = None) -> str:
    concerns = {
        "etf_dividend": [
            "배당률이 높은데 원금 손실은 괜찮은가?",
            "지금 보유해야 하나, 손절해야 하나?",
            "추가 매수하면 배당으로 손실을 회복할 수 있는가?",
            "배당이 실제 수익인지 자본 반환인지 어떻게 구분해야 하는가?",
        ],
        "ai_semiconductor": [
            "이미 많이 오른 AI 반도체 종목을 지금 사도 되는가?",
            "이번 뉴스가 단기 테마인지 장기 성장 동력인지 어떻게 판단해야 하는가?",
            "엔비디아, AMD, 인텔, TSMC, 퀄컴 중 실제 수혜 기업은 어디인가?",
            "데이터센터 외 신규 시장이 실제 실적에 반영될 수 있는가?",
        ],
        "ai_manufacturing": [
            "AI가 반도체 공정 생산성을 실제로 높일 수 있는가?",
            "TSMC 같은 파운드리의 수율 개선이 엔비디아 투자 논리를 강화하는가?",
            "GPU 판매가 아니라 제조공정 AI 확산으로 TAM이 넓어지는가?",
            "제조 CAPEX 절감과 생산성 향상이 실적에 언제 반영될 수 있는가?",
        ],
        "ai_enterprise": [
            "AI Agent가 실제 기업 생산성 향상으로 이어질 수 있는가?",
            "기업 고객이 AI 기능에 추가 비용을 지불할 의지가 있는가?",
            "AI 도입이 클라우드 사용량과 SaaS ARPU를 높일 수 있는가?",
            "보안, 규제, 도입 비용이 확산 속도를 늦출 수 있는가?",
        ],
        "ai_healthcare": [
            "의료 AI Agent가 병원 현장에서 실제로 채택될 수 있는가?",
            "임상의 업무 보조가 생산성 개선과 비용 절감으로 이어지는가?",
            "환자 데이터 보안과 규제 리스크는 어느 정도인가?",
            "헬스케어 AI가 단기 테마인지 장기 산업 침투인지 어떻게 판단할까?",
        ],
        "bigtech": [
            "리스크 뉴스에도 주가가 오르는 이유는 무엇인가?",
            "빅테크의 AI CAPEX가 장기 성장으로 이어질 수 있는가?",
            "규제, 보안, 독점 이슈가 밸류에이션을 훼손할 수 있는가?",
            "이미 보유 중인 빅테크 주식을 계속 가져가도 되는가?",
        ],
        "cashflow_value": [
            "현금이 많은 기업이 정말 좋은 투자처인가?",
            "자유현금흐름이 좋아도 주가가 부진한 이유는 무엇인가?",
            "자사주 매입과 배당 중 무엇이 더 중요한가?",
            "기업의 자본배분 능력을 어떻게 판단해야 하는가?",
        ],
        "consumer_cyclical": [
            "경기 둔화에도 이 기업의 수요가 유지될 수 있는가?",
            "금리와 소비 위축이 실적에 얼마나 영향을 주는가?",
            "브랜드 파워와 가격 전가력이 실제로 작동하는가?",
            "지금은 저가 매수 기회인가, 구조적 둔화의 시작인가?",
        ],
    }
    items = concerns.get(article_type, [
        "이 뉴스가 기존 투자 논리를 강화하는가, 약화하는가?",
        "이미 보유한 투자자는 무엇을 확인해야 하는가?",
        "신규 진입을 고민하는 투자자는 어떤 리스크를 먼저 봐야 하는가?",
    ])
    if secondary_topic and secondary_topic in concerns:
        items = items + concerns[secondary_topic][:2]
    return "\n".join(f"- {item}" for item in items)


def risk_guide_for_topic(article_type: str) -> str:
    risks = {
        "etf_dividend": "- NAV 훼손\n- 배당 착시\n- 총수익률 부진\n- 자본 반환(ROC)\n- 주식 병합",
        "ai_semiconductor": "- 기대가 이미 주가에 반영된 밸류에이션 부담\n- 고객 채택률 부진\n- 데이터센터 외 시장의 실적 반영 지연\n- 경쟁사의 가격 방어 전략\n- 공급망 병목",
        "ai_manufacturing": "- 수율 개선 효과의 실적 반영 지연\n- 제조 현장 도입 속도 부진\n- 파운드리 고객 집중 리스크\n- 공정 자동화 ROI 검증 부족\n- 제조 CAPEX 절감 기대 과대평가",
        "ai_enterprise": "- 기업 고객 도입 지연\n- 보안 비용 증가\n- AI 기능의 가격 전가 실패\n- 기존 워크플로우와의 통합 비용\n- Churn Risk",
        "ai_healthcare": "- 의료 데이터 보안\n- 규제 승인과 책임 소재\n- 병원 시스템 통합 지연\n- 임상의 채택 저항\n- 단기 매출화 지연",
        "bigtech": "- 고객 신뢰 훼손\n- 규제 조사\n- 보안 비용 증가\n- 엔터프라이즈 계약 지연\n- 플랫폼 락인 약화",
        "cashflow_value": "- 낮은 ROIC\n- 비효율적 M&A\n- 성장 없는 현금 보유\n- 자사주 매입 타이밍 실패\n- 경기민감 현금흐름 착시",
        "consumer_cyclical": "- 재고 부담\n- 수요 둔화\n- 금리 민감도\n- 가격 전가력 약화\n- 브랜드 프리미엄 훼손",
    }
    return risks.get(article_type, "- 실적 반영 지연\n- 밸류에이션 부담\n- 경쟁 구도 변화\n- 규제 또는 수요 리스크")


def title_templates_for_topic(article_type: str) -> str:
    templates = {
        "etf_dividend": "- OO ETF 괜찮을까? 배당률보다 먼저 봐야 할 핵심 리스크\n- OO 고배당 ETF, 배당은 높은데 왜 원금이 줄어들까?\n- OO ETF 보유해도 될까? 총수익률과 NAV 훼손 분석",
        "ai_semiconductor": "- OO 신제품이 중요한 이유, 지금 투자자가 확인해야 할 핵심 변수\n- OO 주가 이미 많이 올랐는데 지금 사도 될까? AI 성장성과 리스크 분석\n- OO와 OO 경쟁 구도, 실제 수혜 기업은 어디일까?",
        "ai_manufacturing": "- TSMC가 엔비디아 AI를 공장에 도입한 이유, 반도체 수율 경쟁이 시작됐다\n- AI는 이제 반도체 공장을 운영한다: 엔비디아와 TSMC 협력의 진짜 의미\n- 엔비디아 AI가 GPU 판매를 넘어 제조공정으로 확산되는 이유",
        "ai_enterprise": "- AI Agent는 실제 기업 생산성을 바꿀까? 투자자가 봐야 할 체크포인트\n- OO AI 기능, 단기 테마인가 엔터프라이즈 성장 동력인가?\n- AI 소프트웨어 확산이 클라우드 실적에 반영되는 조건",
        "ai_healthcare": "- 의료 AI Agent는 병원 현장을 바꿀까? 투자자가 봐야 할 리스크\n- AI 헬스케어 시장, 단기 테마와 장기 침투율을 구분하는 법\n- 의료 AI 도입이 실제 매출로 연결되기 위한 조건",
        "bigtech": "- OO 리스크에도 주가가 오른 이유, 시장은 무엇을 보고 있나?\n- OO 보유해도 될까? AI CAPEX와 규제 리스크를 함께 보는 법\n- OO 주가 상승의 진짜 이유, 단기 뉴스보다 중요한 구조적 변수",
        "cashflow_value": "- 현금흐름이 좋은 기업이 반드시 좋은 투자일까? FCF와 자본배분의 함정\n- 현금 많은 기업에 투자하면 안전할까? 기관투자자가 FCF를 보는 이유\n- OO 주식, 현금흐름은 좋은데 왜 시장 평가는 낮을까?",
    }
    return templates.get(article_type, "- OO 뉴스 이후 투자자는 무엇을 판단해야 할까?\n- OO 주식 지금 봐야 할 핵심 리스크와 기회\n- 단기 뉴스보다 중요한 투자 체크포인트")


def image_suggestions_for_topic(article_type: str) -> str:
    suggestions = {
        "etf_dividend": "1. 배당률 vs 총수익률 비교 차트\n2. NAV 추세 이미지\n3. ETF 구조 설명 이미지\n4. 분배금과 자본 반환(ROC) 구조도",
        "ai_semiconductor": "1. AI 반도체 공급망 구조도\n2. 엔비디아/AMD/인텔/TSMC 경쟁 구도\n3. 데이터센터 외 TAM 확장 지도\n4. AI 칩 밸류체인 이미지",
        "ai_manufacturing": "1. 반도체 공정 AI 적용 구조도\n2. Lithography/수율/결함검사 흐름도\n3. TSMC-엔비디아 제조공정 협력 지도\n4. 제조 CAPEX 절감 구조도",
        "ai_enterprise": "1. AI Agent 업무 자동화 흐름도\n2. 클라우드/SaaS 수익화 구조도\n3. 기업 고객 도입 리스크 맵",
        "ai_healthcare": "1. 의료 AI Agent 워크플로우\n2. 병원/임상의/환자 데이터 흐름도\n3. 헬스케어 AI 규제 리스크 맵",
        "bigtech": "1. 플랫폼 구조도\n2. AI CAPEX와 수익화 연결 지도\n3. 규제/보안 리스크 맵",
        "cashflow_value": "1. FCF 사용처 흐름도\n2. 자본배분 구조도\n3. 자사주 매입과 배당 비교 이미지",
    }
    return suggestions.get(article_type, "1. 투자 리스크 구조도\n2. 경쟁 구도 표 이미지\n3. 핵심 체크리스트 이미지")


def cony_style_instruction(article_type: str) -> str:
    if article_type != "etf_dividend":
        return ""
    return """
ETF/배당/고위험 인컴 상품은 CONY형 구조를 우선 적용하세요:
- [경험/문제 제기] 왜 이 상품이 관심을 받는가?
- [본질 정의] 이 상품의 진짜 정체는 무엇인가?
- [겉으로 보이는 매력] 배당, 분배금, 월수입, 고수익률
- [숨겨진 리스크] NAV 훼손, ROC, 주식 병합, 총수익률 부진
- [투자자별 적합성] 누구에게 맞고 누구에게 맞지 않는가?
- [행동 기준] 보유 / 신규 진입 / 추가매수 / 손절 기준
- [공식 자료 링크]
""".strip()


def build_manual_gpt_prompt(article: NewsArticle, history: dict, tickers: list[str]) -> str:
    related_posts = find_related_posts(history, {"category": "미국증시"}, tickers)
    related_text = "\n".join(
        f"- {post['title']}: {post['url']}" for post in related_posts
    ) or "- 제공된 과거 글 없음. 아래 카테고리 링크 대체 규칙을 사용하세요."
    limited_source = is_limited_source(article)
    quality_score = content_quality_score(article)
    primary_topic, secondary_topic = classify_article_topics(article, tickers)
    primary_topic_name = article_type_label(primary_topic)
    secondary_topic_name = article_type_label(secondary_topic) if secondary_topic else "없음"
    guide = article_type_guide(primary_topic)
    if secondary_topic:
        guide = f"{guide}\n\n보조 주제 확장 가이드:\n{article_type_guide(secondary_topic)}"
    fallback_links = fallback_related_links_html(primary_topic)
    concerns = investor_concerns_for_topic(primary_topic, secondary_topic)
    risk_guide = risk_guide_for_topic(primary_topic)
    title_templates = title_templates_for_topic(primary_topic)
    image_suggestions = image_suggestions_for_topic(primary_topic)
    cony_instruction = cony_style_instruction(primary_topic)
    limited_source_instruction = (
        """- 제한적 원문 여부: True
- 원문이 Premium/부분 공개 기사이거나 본문 데이터가 부족합니다.
- 얇은 기사 요약문으로 작성하지 마세요. 제목과 공개 요약은 "시드 이슈"로만 사용하세요.
- 본문에서 방어 문구를 반복하지 말고, 데이터 한계는 마지막 투자 책임 고지 문단에서 1회만 언급하세요.
- 전체 글 분량 비율은 기사 요약 10% 이하, 산업 구조 분석 40%, 투자 프레임워크 30%, Bull/Bear/Mispricing 20%로 작성하세요.
- 확인되지 않은 실적 수치, 목표주가, PER/PBR, EPS, 컨센서스, 애널리스트 의견, 성능 데이터는 절대 만들지 마세요.
- 일반 기업 소개로 분량을 채우지 말고, primary_topic과 secondary_topic에 맞는 독자 고민, 산업 침투율, 생산성, 비용 구조, 규제 리스크, TAM 확장 프레임워크로 확장하세요."""
        if limited_source
        else
        """- 제한적 원문 여부: False
- 원문 본문이 충분하더라도 기사에 없는 재무 수치, 목표주가, 컨센서스, 공시 내용은 만들지 마세요.
- 단순 요약이 아니라 투자자 관점의 시장 기대치와 mispricing 가능성을 분석하세요.
- 데이터 한계 문구는 필요할 때 마지막 투자 책임 고지 문단에서만 1회 사용하세요."""
    )

    return f"""
당신은 기관투자자(Buy-side) 스타일의 퀀트 기반 재무분석가이자 10년 차 자산운용사 매니저입니다.

당신의 글은 단순 뉴스 해설이 아니라, 이 뉴스를 검색한 개인 투자자의 실제 고민을 해결하는 투자 판단형 콘텐츠여야 합니다.
모든 글은 다음 질문에 답해야 합니다.
- 이미 이 종목을 보유한 투자자는 무엇을 해야 하는가?
- 신규 진입을 고민하는 투자자는 무엇을 확인해야 하는가?
- 단기 트레이더와 장기 투자자의 관점은 어떻게 다른가?
- 이 뉴스가 기존 투자 논리를 강화하는가, 약화하는가?

목표:
- 아래 기사 데이터를 바탕으로 구글 검색 상위 노출을 노리는 한국어 Blogger/Tistory용 HTML 글을 작성하세요.
- 기존 흐름인 "뉴스 발생 → 산업 구조 분석"에 머물지 말고, "독자의 투자 고민 → 뉴스의 의미 → 구조 분석 → 투자자별 행동 기준 → 체크리스트" 순서로 작성하세요.
- 수집되지 않은 재무 수치, 컨센서스, 목표주가, 애널리스트 의견, 공시 내용은 절대 지어내지 마세요.
- AI 기사는 GPU/데이터센터 일반론으로 자동 확장하지 말고, primary_topic과 secondary_topic이 가리키는 실제 침투 영역을 중심으로 작성하세요.
- 인사말, 감탄사, "첫째로", "요약하자면" 같은 진부한 표현 없이 바로 분석으로 들어가세요.

제한적 원문 처리:
{limited_source_instruction}
- 글 길이는 공백 제외 최소 2,000자 이상, 가능하면 2,500~3,500자 수준으로 작성하세요.

이 글을 검색한 독자의 핵심 고민:
{concerns}

도입부 작성 규칙:
- 도입부는 산업 해설이 아니라 이미 보유한 투자자, 신규 진입 투자자, 손실 중인 투자자, 고점 추격 매수 투자자, 배당/테마 수익 기대 투자자의 고민 중 하나로 시작하세요.
- 예: 이미 NVDA를 보유한 투자자라면 이번 뉴스를 단순 신제품 뉴스로 볼지, 데이터센터 의존도를 낮추는 장기 성장 옵션으로 볼지 고민할 수 있습니다.
- 예: 고배당 ETF를 보유한 투자자라면 매달 들어오는 분배금이 실제 수익인지, 원금 훼손을 가리는 착시인지 반드시 구분해야 합니다.

제목 생성 규칙:
- 제목은 "무슨 일이 있었나?"보다 "투자자는 무엇을 판단해야 하나?"에 가깝게 작성하세요.
- 뉴스 제목 번역형 제목을 피하고, 투자자 검색 질문형 제목을 우선하세요.
기사 유형별 제목 템플릿:
{title_templates}

기사 유형별 확장 가이드:
{guide}

기사 유형별 리스크 구체화:
{risk_guide}

{cony_instruction}

개인 투자자 접근법 작성 규칙:
- 반드시 보유자, 신규 진입자, 단기 트레이더, 장기 투자자를 분리하세요.
- 기사 유형별로 문장을 구체화하세요.
- ETF/배당 기사라면 총수익률, NAV 훼손, ROC, 추가매수/손절 기준을 우선하세요.
- AI/반도체 기사라면 신제품 발표보다 실제 고객 채택률, 실적 기여 시점, 밸류에이션 부담을 우선하세요.
- 빅테크 보안 기사라면 보안 비용, 고객 신뢰, 규제 리스크, 플랫폼 락인을 우선하세요.

투자 판단 체크리스트 작성 규칙:
- 체크리스트는 기사 유형별로 구체화하세요.
- CONY/ETF형 글은 총수익률, NAV 추세, ROC, 주식 병합, 분배금 지속 가능성을 포함하세요.
- AI PC/AI 반도체형 글은 고객 채택률, 실적 기여 시점, 경쟁사 대응, 교체 수요, 밸류에이션 부담을 포함하세요.
- 현금흐름/가치주형 글은 FCF, ROIC, 자본배분, 자사주 매입 타이밍, 성장성 부족 리스크를 포함하세요.

가독성 및 리스트 작성 규칙:
- <li> 항목은 평문으로 시작하지 말고 반드시 <strong>핵심 키워드</strong>: 형식으로 시작하세요.
- 경쟁 구도, 구조적 변수, 투자 체크포인트처럼 순서와 인과관계가 중요한 영역은 <ul>보다 <ol>을 우선 사용하세요.

내부 링크 지침:
- 관련 과거 글이 있으면 본문 중간 문맥에 자연스러운 앵커 텍스트 링크로 최소 1개 이상 삽입하세요.
- 제공된 과거 글이 없으면 부재 안내 문장을 쓰지 말고 아래 카테고리 링크를 사용하세요.

카테고리 대체 링크:
<h2>📚 함께 보면 좋은 글</h2>
<ul>
{fallback_links}
</ul>

FAQ 스니펫 지침:
- FAQ 질문은 <h3> 태그로 작성하세요.
- 첫 번째 답변 문단은 바로 다음 <p><strong>정답 요약:</strong> ...</p>에 배치하세요.
- 정답 요약 문단은 정확히 1~2문장, 가능하면 30단어 이하로 직접적인 결론만 담으세요.
- 보충 논리, 배경, 추적 지표는 반드시 바로 아래 별도 <p> 문단으로 분리하세요.

출력:
- Blogger/Tistory에 바로 붙여넣을 수 있는 HTML만 출력하세요.
- Markdown 설명, 코드블록, 별도 해설은 출력하지 마세요.
- 아래 HTML 구조의 모든 섹션을 생략하지 마세요.

HTML 구조:
<h1>SEO 제목</h1>
<p>이 글을 검색한 투자자의 고민과 뉴스의 의미를 연결한 도입부 2~3문장</p>
<hr>
<blockquote style="background: #f9f9f9; border-left: 8px solid #007bff; padding: 15px; margin: 20px 0;">
  📌 <strong>기관 투자자 관점 핵심 3줄 요약</strong><br>
  • 핵심 포인트 1<br>
  • 핵심 포인트 2<br>
  • 핵심 포인트 3
</blockquote>
<h2>🔎 이 글을 검색한 투자자의 핵심 고민</h2>
<ul>
  <li><strong>고민 1</strong>: 기사 유형에 맞는 실제 투자 고민</li>
  <li><strong>고민 2</strong>: 보유/신규진입/손절/추가매수 관련 고민</li>
  <li><strong>고민 3</strong>: 리스크 또는 기회 관련 고민</li>
</ul>
<h2>📊 시장 기대치와의 괴리 (Expectation Gap)</h2>
<p>시장 기대치와 실제 뉴스 사이의 차이를 Fact 중심으로 분석</p>
<h2>🏢 기업의 현재 위치와 경쟁 구도</h2>
<p>분석 대상 기업이 현재 산업 안에서 어떤 위치에 있는지, 관련 경쟁 구도와 함께 설명</p>
<h2>⚙️ 기술 전략과 원가 구조의 의미 (Technical Strategy & Cost Structure)</h2>
<ol>
  <li><strong>구조적 비용 변수</strong>: 기사 유형에 맞는 비용 또는 효율성 레버를 설명</li>
  <li><strong>마진 민감도</strong>: 수율, 자동화, 기술 부채, 고객 이탈률, CAPEX 감가상각 등 마진에 영향을 주는 변수를 설명</li>
  <li><strong>투자자 체크포인트</strong>: 향후 실적 또는 공시에서 추적할 지표를 설명</li>
</ol>
<h2>🧭 개인 투자자는 어떻게 접근해야 할까?</h2>
<ul>
  <li><strong>이미 보유한 투자자</strong>: 이번 뉴스가 기존 투자 논리를 강화하는지, 약화하는지 판단한다.</li>
  <li><strong>신규 진입을 고민하는 투자자</strong>: 뉴스 직후 추격 매수보다 실적 반영 가능성과 밸류에이션 부담을 함께 확인한다.</li>
  <li><strong>단기 트레이더</strong>: 뉴스 모멘텀, 수급, 기대감 반영 속도를 중심으로 접근한다.</li>
  <li><strong>장기 투자자</strong>: 구조적 성장 시장인지 일시적 테마인지 구분한다.</li>
</ul>
<h2>❓ 무엇이 가장 중요할까요? (FAQ)</h2>
<h3>개인 투자자가 검색할 만한 핵심 질문</h3>
<p><strong>정답 요약:</strong> 구글 추천 스니펫에 적합하도록 1~2문장으로 결론만 답변</p>
<p>보충 논리와 추적해야 할 지표를 별도 문단으로 설명</p>
<h2>⚖️ 찬반 논리 점검 (Bull vs Bear Thesis)</h2>
<ul>
  <li><strong>상방 모멘텀 (Bull)</strong>: 성장 모멘텀 또는 catalyst 중심 긍정 논리</li>
  <li><strong>하방 리스크 (Bear)</strong>: 밸류에이션, 희석, 규제, 수요 둔화 등 반대 논리</li>
</ul>
<h2>💡 월가의 오판 포인트 (Market Mispricing)</h2>
<p style="color: #2c3e50; font-weight: bold; background: #f0f7ff; padding: 12px; border-radius: 5px;">현재 시장의 가정 → 실제 가능성 → 맞을 경우 수혜 → 틀릴 경우 리스크 구조로 작성</p>
<h2>✅ 투자 판단 전 체크리스트</h2>
<ul>
  <li><strong>체크포인트 1</strong>: 기사 유형별 핵심 확인 사항</li>
  <li><strong>체크포인트 2</strong>: 실적 또는 공시에서 확인할 사항</li>
  <li><strong>체크포인트 3</strong>: 리스크 관리 기준</li>
</ul>
<h2>📚 함께 보면 좋은 글</h2>
<ul>
  관련 과거 글이 있으면 자연스러운 앵커 링크 삽입
  제공된 과거 글이 없으면 카테고리 대체 링크 사용
</ul>
<div style="font-size: 0.9em; color: #6b7280; margin-top: 30px; text-align: right;">
  원본 출처: <a href="{html.escape(article.url)}" target="_blank" rel="noopener noreferrer nofollow">{html.escape(article.source_name)}</a>
</div>
<br>
<div style="background: #f8fafc; color: #64748b; font-size: 0.8em; line-height: 1.5; text-align: center; padding: 12px; border-radius: 6px; margin-top: 12px;">
  ※ 본 분석은 의사결정 참고용 데이터입니다. 일부 원문 데이터가 제한적인 경우 최종 투자 판단 전 공식 공시와 실적 자료 확인이 필요하며, 최종 투자 책임은 사용자 본인에게 있습니다.
</div>
<!-- 이미지 제안:
{image_suggestions}
-->

기사 데이터:
- 제목: {article.title}
- 출처: {article.source_name}
- URL: {article.url}
- 감지된 티커 후보: {tickers}
- RSS 요약: {article.rss_summary[:700]}
- 제한적 원문 여부: {limited_source}
- 본문 품질 점수: {quality_score}
- primary_topic: {primary_topic}
- primary_topic_name: {primary_topic_name}
- secondary_topic: {secondary_topic or "없음"}
- secondary_topic_name: {secondary_topic_name}
- 전처리된 본문: {article.clean_text}

관련 과거 글:
{related_text}
""".strip()


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
