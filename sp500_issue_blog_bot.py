from __future__ import annotations

import html
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import schedule
import yfinance as yf
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI


KST_RUN_TIME = "07:30"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BLOGGER_SCOPES = ["https://www.googleapis.com/auth/blogger"]
MAJOR_BANKS = (
    "JPMorgan",
    "JP Morgan",
    "Goldman",
    "Morgan Stanley",
    "Bank of America",
    "BofA",
    "Citigroup",
    "Citi",
    "Wells Fargo",
    "Barclays",
    "Deutsche Bank",
    "UBS",
    "Jefferies",
    "Bernstein",
    "RBC",
    "Evercore",
    "Mizuho",
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


@dataclass
class CompanyIssue:
    ticker: str
    company_name: str
    current_price: float | None
    daily_change_pct: float | None
    issue_type: str
    issue_summary: str
    logo_url: str | None


def clean_ticker_for_yfinance(ticker: str) -> str:
    return ticker.replace(".", "-").strip()


def get_sp500_tickers(limit: int | None = None) -> list[str]:
    tables = pd.read_html(SP500_WIKI_URL)
    constituents = tables[0]
    tickers = [clean_ticker_for_yfinance(t) for t in constituents["Symbol"].tolist()]
    return tickers[:limit] if limit else tickers


def safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_logo_url(info: dict) -> str | None:
    logo_url = info.get("logo_url")
    if logo_url:
        return logo_url

    website = info.get("website")
    if not website:
        return None

    domain = urlparse(website).netloc.replace("www.", "")
    if not domain:
        return None
    return f"https://logo.clearbit.com/{domain}"


def summarize_price(info: dict, history: pd.DataFrame) -> tuple[float | None, float | None]:
    current_price = safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("postMarketPrice")
        or info.get("previousClose")
    )

    if current_price is None and not history.empty:
        current_price = safe_float(history["Close"].iloc[-1])

    previous_close = safe_float(info.get("previousClose"))
    if previous_close is None and len(history.index) >= 2:
        previous_close = safe_float(history["Close"].iloc[-2])

    if current_price is None or not previous_close:
        return current_price, None

    return current_price, round(((current_price - previous_close) / previous_close) * 100, 2)


def is_recent_earnings_event(ticker_obj: yf.Ticker, today_utc: date) -> str | None:
    start = pd.Timestamp(today_utc - timedelta(days=1), tz="UTC")
    end = pd.Timestamp(today_utc + timedelta(days=1), tz="UTC")

    try:
        earnings_dates = ticker_obj.get_earnings_dates(limit=12)
    except Exception as exc:
        logging.debug("Earnings lookup failed: %s", exc)
        earnings_dates = None

    if earnings_dates is not None and not earnings_dates.empty:
        idx = earnings_dates.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        recent = earnings_dates[(idx >= start) & (idx <= end)]
        if not recent.empty:
            row = recent.iloc[0]
            eps = safe_float(row.get("Reported EPS"))
            surprise = safe_float(row.get("Surprise(%)"))
            parts = ["Earnings released today or after the prior close"]
            if eps is not None:
                parts.append(f"reported EPS {eps:g}")
            if surprise is not None:
                parts.append(f"surprise {surprise:g}%")
            return ", ".join(parts)

    try:
        calendar = ticker_obj.calendar
        earnings_date = calendar.get("Earnings Date") if isinstance(calendar, dict) else None
        if isinstance(earnings_date, list):
            earnings_date = earnings_date[0]
        if earnings_date:
            earnings_day = pd.Timestamp(earnings_date).date()
            if today_utc - timedelta(days=1) <= earnings_day <= today_utc:
                return f"Earnings date confirmed for {earnings_day.isoformat()}"
    except Exception as exc:
        logging.debug("Calendar lookup failed: %s", exc)

    return None


def is_recent_rating_event(ticker_obj: yf.Ticker, today_utc: date) -> str | None:
    try:
        upgrades = ticker_obj.upgrades_downgrades
    except Exception as exc:
        logging.debug("Rating lookup failed: %s", exc)
        upgrades = None

    if upgrades is not None and not upgrades.empty:
        data = upgrades.copy()
        idx = pd.to_datetime(data.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        data["_event_date"] = idx.date
        recent = data[
            data["_event_date"].isin({today_utc, today_utc - timedelta(days=1)})
        ].copy()

        if not recent.empty and "Firm" in recent:
            firm_filter = recent["Firm"].astype(str).str.contains(
                "|".join(re.escape(firm) for firm in MAJOR_BANKS),
                case=False,
                na=False,
            )
            recent = recent[firm_filter]

        if not recent.empty:
            row = recent.iloc[0]
            firm = row.get("Firm", "Major brokerage")
            action = row.get("Action", "rating action")
            from_grade = row.get("FromGrade", "")
            to_grade = row.get("ToGrade", "")
            grade_text = f"{from_grade} -> {to_grade}".strip(" ->")
            return f"{firm} {action} {grade_text}".strip()

    try:
        for item in ticker_obj.news[:10]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            published = item.get("providerPublishTime")
            if not published:
                continue
            published_date = datetime.utcfromtimestamp(published).date()
            is_recent = published_date in {today_utc, today_utc - timedelta(days=1)}
            mentions_target = re.search(r"price target|upgrade|downgrade|raises|cuts", title, re.I)
            mentions_major_bank = any(bank.lower() in f"{title} {publisher}".lower() for bank in MAJOR_BANKS)
            if is_recent and mentions_target and mentions_major_bank:
                return f"{publisher}: {title}"
    except Exception as exc:
        logging.debug("News lookup failed: %s", exc)

    return None


def collect_sp500_issues(max_companies: int = 3, scan_limit: int | None = None) -> list[CompanyIssue]:
    today_utc = datetime.now(ZoneInfo("America/New_York")).date()
    issues: list[CompanyIssue] = []

    for ticker in get_sp500_tickers(limit=scan_limit):
        if len(issues) >= max_companies:
            break

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info or {}
            history = ticker_obj.history(period="5d", interval="1d", auto_adjust=False)
            current_price, daily_change_pct = summarize_price(info, history)

            earnings_summary = is_recent_earnings_event(ticker_obj, today_utc)
            rating_summary = is_recent_rating_event(ticker_obj, today_utc)

            if not earnings_summary and not rating_summary:
                continue

            issue_type = "Earnings release" if earnings_summary else "Analyst rating or price-target change"
            issue_summary = earnings_summary or rating_summary or "Major market-moving issue confirmed"
            company_name = info.get("longName") or info.get("shortName") or ticker

            issues.append(
                CompanyIssue(
                    ticker=ticker,
                    company_name=company_name,
                    current_price=current_price,
                    daily_change_pct=daily_change_pct,
                    issue_type=issue_type,
                    issue_summary=issue_summary,
                    logo_url=get_logo_url(info),
                )
            )
            logging.info("Selected %s: %s", ticker, issue_summary)
        except Exception as exc:
            logging.warning("Skipping %s: %s", ticker, exc)

    return sorted(
        issues,
        key=lambda x: abs(x.daily_change_pct or 0),
        reverse=True,
    )[:max_companies]


def build_prompt(issues: Iterable[CompanyIssue]) -> str:
    issue_dicts = [asdict(issue) for issue in issues]
    return f"""
You are a 10-year asset-management portfolio manager who explains US stock-market trends in a clear, lively, and useful way.
Write an English SEO-optimized blog post in Markdown for a Google Blogger audience.
Use a professional, confident advisor tone. Keep the article readable and energetic with relevant emojis.

Use the S&P 500 issue-company data below.
At the start of each company section, insert the collected logo_url as a raw HTML image tag.
Example:
<img src="LOGO_URL" alt="Company Name logo" width="120" style="margin-bottom:10px;"><br>

Structure:
1. Title: combine a high-search-volume company name with curiosity, e.g. "Tesla Price Target Jumps: Why Wall Street Is Moving Again"
2. Opening: summarize today's hottest S&P 500 earnings or analyst-rating stories.
3. Deep dives by company: each company must include Fact, Wall Street View, and Investor Action.
4. Closing: tomorrow's market watch points and a natural comment prompt.

Constraints:
- Avoid stale phrases such as "firstly", "in conclusion", and "to summarize".
- Do not sound like generic AI copy.
- Write at least 1,500 English words excluding spaces.
- Do not present this as personalized investment advice. Include risk controls and scenario thinking.
- Return Markdown only.

Data:
{issue_dicts}
""".strip()


def generate_blog_post(issues: list[CompanyIssue]) -> str:
    if not issues:
        raise RuntimeError("오늘 조건에 맞는 S&P 500 이슈 기업을 찾지 못했습니다.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {
                "role": "system",
                "content": "You write polished English financial blog posts with SEO-aware structure.",
            },
            {"role": "user", "content": build_prompt(issues)},
        ],
        temperature=0.75,
    )
    return response.choices[0].message.content.strip()


def markdown_to_blogger_html(markdown_text: str) -> str:
    try:
        import markdown

        return markdown.markdown(markdown_text, extensions=["extra", "sane_lists"])
    except ImportError:
        return markdown_text


def extract_title(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return f"Today's S&P 500 Market-Moving Issues {datetime.now().strftime('%Y-%m-%d')}"


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


def upload_to_blogger(markdown_text: str) -> dict:
    blog_id = os.environ["BLOGGER_BLOG_ID"]
    title = extract_title(markdown_text)
    html_content = markdown_to_blogger_html(markdown_text)
    labels = [
        label.strip()
        for label in os.getenv("BLOGGER_LABELS", "S&P 500,US Stocks,Wall Street,Earnings").split(",")
        if label.strip()
    ]
    is_draft = os.getenv("PUBLISH_STATUS", "draft").lower() != "publish"

    body = {
        "kind": "blogger#post",
        "title": title,
        "content": html_content,
        "labels": labels,
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


def upload_blog_post(markdown_text: str) -> dict:
    platform = os.getenv("BLOG_PLATFORM", "blogger").lower()

    if platform == "blogger":
        return upload_to_blogger(markdown_text)

    output_path = os.getenv("LOCAL_OUTPUT_PATH", "latest_sp500_issue_post.md")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown_text)
    return {"status": "saved_local", "path": output_path}


def run_once() -> None:
    load_dotenv()
    max_companies = int(os.getenv("MAX_COMPANIES", "3"))
    scan_limit = os.getenv("SCAN_LIMIT")
    scan_limit_value = int(scan_limit) if scan_limit else None
    skip_if_no_issues = os.getenv("SKIP_IF_NO_ISSUES", "true").lower() == "true"

    issues = collect_sp500_issues(
        max_companies=max_companies,
        scan_limit=scan_limit_value,
    )
    logging.info("Collected %s issue companies", len(issues))

    if not issues and skip_if_no_issues:
        logging.warning("No qualifying issue companies found. Skipping post generation.")
        return

    post = generate_blog_post(issues)
    result = upload_blog_post(post)
    logging.info("Upload result: %s", html.escape(str(result))[:500])


def run_scheduler() -> None:
    load_dotenv()
    schedule.every().day.at(os.getenv("RUN_TIME_KST", KST_RUN_TIME)).do(run_once)
    logging.info("Scheduler started. Daily run time: %s KST", os.getenv("RUN_TIME_KST", KST_RUN_TIME))

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    if os.getenv("RUN_NOW", "false").lower() == "true":
        run_once()
    else:
        run_scheduler()
