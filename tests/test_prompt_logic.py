import importlib
import sys
import types
import unittest
from pathlib import Path


def install_import_stubs() -> None:
    stubs = {
        "feedparser": types.ModuleType("feedparser"),
        "requests": types.ModuleType("requests"),
        "schedule": types.ModuleType("schedule"),
        "bs4": types.ModuleType("bs4"),
        "dotenv": types.ModuleType("dotenv"),
        "google": types.ModuleType("google"),
        "google.auth": types.ModuleType("google.auth"),
        "google.auth.transport": types.ModuleType("google.auth.transport"),
        "google.auth.transport.requests": types.ModuleType("google.auth.transport.requests"),
        "google.oauth2": types.ModuleType("google.oauth2"),
        "google.oauth2.credentials": types.ModuleType("google.oauth2.credentials"),
        "google_auth_oauthlib": types.ModuleType("google_auth_oauthlib"),
        "google_auth_oauthlib.flow": types.ModuleType("google_auth_oauthlib.flow"),
        "googleapiclient": types.ModuleType("googleapiclient"),
        "googleapiclient.discovery": types.ModuleType("googleapiclient.discovery"),
        "openai": types.ModuleType("openai"),
        "zoneinfo": types.ModuleType("zoneinfo"),
    }
    stubs["feedparser"].parse = lambda *_args, **_kwargs: types.SimpleNamespace(feed={}, entries=[])
    stubs["bs4"].BeautifulSoup = object
    stubs["dotenv"].load_dotenv = lambda *_args, **_kwargs: None
    stubs["google.auth.transport.requests"].Request = object
    stubs["google.oauth2.credentials"].Credentials = object
    stubs["google_auth_oauthlib.flow"].InstalledAppFlow = object
    stubs["googleapiclient.discovery"].build = lambda *_args, **_kwargs: None
    stubs["openai"].APIStatusError = Exception
    stubs["openai"].OpenAI = object
    stubs["openai"].RateLimitError = Exception
    stubs["zoneinfo"].ZoneInfo = lambda _key: None

    for name, module in stubs.items():
        sys.modules.setdefault(name, module)


install_import_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
bot = importlib.import_module("sp500_issue_blog_bot")


def article(title: str, summary: str, text: str, source: str = "Test Source") -> bot.NewsArticle:
    return bot.NewsArticle(
        guid=title,
        url="https://example.com/article",
        title=title,
        source_name=source,
        published="",
        rss_summary=summary,
        clean_text=text,
    )


class PromptLogicTests(unittest.TestCase):
    def test_cash_producing_article_prioritizes_cashflow_and_marks_limited(self):
        sample = article(
            "1 Cash-Producing Stock to Research Further and 2 Facing Challenges",
            "Generating cash is essential for any business, but not all cash-rich companies are great investments. "
            "Some produce plenty of cash but fail to allocate it effectively.",
            "Oops, something went wrong Skip to navigation More from Yahoo Scout "
            "How do these companies' cash flow margins compare? "
            "What makes Apple a compelling cash-producing investment?",
            "Yahoo! Finance: AAPL MSFT NVDA TSLA AMD META GOOGL AMZN News",
        )

        self.assertEqual(bot.classify_article_type(sample, ["AAPL", "AMD", "NVDA"]), "cashflow_value")
        self.assertTrue(bot.is_limited_source(sample))
        self.assertLessEqual(bot.content_quality_score(sample), 45)

    def test_intel_ai_chip_article_classifies_as_ai_semiconductor(self):
        sample = article(
            "Intel Plans AI Chip Using Cheaper Memory by Year-end",
            "Intel plans AI chip using cheaper memory.",
            "Intel unveiled an inference GPU and plans to launch AI chips using lower-cost memory for data center customers.",
        )

        self.assertEqual(bot.classify_article_type(sample, ["INTC"]), "ai_semiconductor")

    def test_semiconductor_context_can_outweigh_single_cashflow_angle(self):
        sample = article(
            "Nvidia Free Cash Flow Rises as AI GPU Demand Expands",
            "Nvidia's data center GPU demand and AI accelerator cycle remain central to the investment debate.",
            "The semiconductor leader is tied to AI training, inference, HBM memory, data center capex and GPU supply. "
            "Free cash flow and capital allocation matter, but the article's core context is the AI chip cycle.",
        )

        self.assertEqual(bot.classify_article_type(sample, ["NVDA"]), "ai_semiconductor")

    def test_etf_dividend_article_classifies_as_etf_dividend(self):
        sample = article(
            "High Yield ETF Dividend Risk",
            "The ETF offers high distribution yield but faces NAV erosion risk.",
            "Covered call ETF investors should focus on total return, NAV erosion, distribution sustainability and expense ratio.",
        )

        self.assertEqual(bot.classify_article_type(sample, []), "etf_dividend")

    def test_manual_prompt_uses_non_defensive_seo_structure(self):
        sample = article(
            "Microsoft Security Issue Raises Enterprise Risk Questions",
            "A security issue has raised questions about enterprise software risk.",
            "Premium Sign in Skip to navigation Enterprise customers are watching cloud security and software reliability.",
        )
        prompt = bot.build_manual_gpt_prompt(sample, {"items": []}, ["MSFT"])

        self.assertIn("⚙️ 기술 전략과 원가 구조의 의미", prompt)
        self.assertIn("데이터 한계는 마지막 투자 책임 고지 문단에서 1회만", prompt)
        self.assertIn("카테고리 대체 링크", prompt)
        self.assertIn("rel=\"noopener noreferrer nofollow\"", prompt)
        self.assertNotIn("관련 과거 글은 아직 없습니다", prompt)


if __name__ == "__main__":
    unittest.main()
