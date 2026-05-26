# S&P 500 Issue Blog Bot

매일 한국 시간 오전 7시 30분에 S&P 500 기업 중 실적 발표 또는 투자의견/목표주가 변경 이슈가 있는 2~3개 기업을 추려 OpenAI API로 영문 SEO형 글을 만들고 Google Blogger에 업로드합니다.

## 1. 로컬 설치

```bash
pip install -r requirements.txt
```

## 2. Blogger API 준비

Google Cloud Console에서 Blogger API를 활성화한 뒤 OAuth Client JSON을 내려받아 `client_secret.json`으로 저장하세요.

로컬에서 한 번 실행해 Google 로그인을 완료하면 `token.json`이 생성됩니다.

```bash
RUN_NOW=true python sp500_issue_blog_bot.py
```

GitHub Actions는 브라우저 로그인을 할 수 없기 때문에, 이 `token.json`을 GitHub Secret으로 저장해야 합니다.

## 3. GitHub Secrets

GitHub 저장소에서 `Settings` -> `Secrets and variables` -> `Actions`로 이동해 아래 Secrets를 추가하세요.

```text
OPENAI_API_KEY
BLOGGER_BLOG_ID
GOOGLE_CLIENT_SECRET_JSON
GOOGLE_TOKEN_JSON
```

`GOOGLE_CLIENT_SECRET_JSON`에는 `client_secret.json` 전체 내용을 넣고, `GOOGLE_TOKEN_JSON`에는 `token.json` 전체 내용을 넣습니다.

## 4. GitHub Variables

같은 화면의 `Variables` 탭에는 아래 값을 필요에 따라 추가하세요. 추가하지 않으면 워크플로우 기본값을 사용합니다.

```text
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.45
PUBLISH_STATUS=draft
MAX_COMPANIES=3
SCAN_LIMIT=
SKIP_IF_NO_ISSUES=true
AUTO_GENERATE_LABELS=true
MAX_BLOGGER_LABELS=10
BLOGGER_LABELS=S&P 500,US Stocks,Wall Street,Earnings
```

처음에는 `PUBLISH_STATUS=draft`로 며칠 확인한 뒤, 자동 발행이 충분히 안정적이면 `publish`로 바꾸는 편이 좋습니다.

`AUTO_GENERATE_LABELS=true`이면 글 내용과 수집된 기업 데이터를 바탕으로 SEO 라벨을 자동 생성합니다. `BLOGGER_LABELS`는 자동 라벨 생성에 실패하거나 `AUTO_GENERATE_LABELS=false`일 때만 사용됩니다.

본문 작성 프롬프트는 수집 데이터에 없는 실적 수치, 목표주가, 애널리스트 이름, 은행명, 전망을 임의로 만들지 않도록 제한되어 있습니다. 데이터에 없는 내용은 확인 필요 또는 시나리오 관점으로만 다루게 됩니다.

## 5. 자동 실행

워크플로우 파일은 `.github/workflows/blogger-auto-post.yml`에 들어 있습니다.

자동 실행 시간:

```text
매일 한국 시간 07:30
```

GitHub Actions의 cron은 UTC 기준이라 워크플로우에는 `30 22 * * *`로 설정되어 있습니다.

수동 실행은 GitHub 저장소의 `Actions` 탭에서 `Blogger Auto Post`를 선택한 뒤 `Run workflow`를 누르면 됩니다.

## 6. 로컬 실행

한 번만 즉시 실행:

```bash
RUN_NOW=true python sp500_issue_blog_bot.py
```

로컬에서 계속 켜두고 스케줄러 실행:

```bash
python sp500_issue_blog_bot.py
```

업로드 없이 파일로만 확인하려면 `.env`에서 `BLOG_PLATFORM=local`을 사용하세요.
