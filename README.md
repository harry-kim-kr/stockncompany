# RSS Blogger Auto Post Bot

월-금 한국 시간 오전 7시 전후에 RSS 뉴스를 수집하고, 중복을 먼저 차단한 뒤, `gpt-4o-mini` 1회 호출로 기관 투자자 관점의 SEO 재무분석 JSON을 받아 Python HTML 템플릿과 조립해 Google Blogger에 업로드합니다.

## 핵심 구조

- 하루 최대 1개 포스팅: `MAX_POSTS_PER_RUN=1`
- 저비용 모델 고정: `gpt-4o-mini`
- 기사 1건당 OpenAI API 최대 1회 호출
- OpenAI는 긴 HTML을 만들지 않고 JSON만 반환
- Python이 Expectation Gap, Bull/Bear Thesis, Market Mispricing, FAQ, 내부 링크가 포함된 최종 HTML 템플릿을 조립
- 중복 기준: RSS GUID, 기사 URL, 기사 제목 중 하나라도 일치하면 API 호출 전 스킵
- 전처리 후 본문이 400자 미만이거나 부실하면 OpenAI 호출 전 스킵
- `history.json` 기반으로 같은 카테고리 또는 같은 티커의 과거 글을 본문 하단에 자동 연결
- 발행 성공 후 `history.json`, `run_summary.json`을 GitHub Actions가 자동 Commit & Push
- `MANUAL_PROMPT_MODE=true`이면 OpenAI API와 Blogger API를 호출하지 않고 ChatGPT 웹에 붙여넣을 프롬프트만 생성

## 설치

```bash
pip install -r requirements.txt
```

## GitHub Secrets

GitHub 저장소 `Settings` -> `Secrets and variables` -> `Actions`에서 아래 Secrets를 추가하세요.

```text
OPENAI_API_KEY
BLOGGER_BLOG_ID
GOOGLE_CLIENT_SECRET_JSON
GOOGLE_TOKEN_JSON
```

## GitHub Variables

필수:

```text
RSS_FEED_URLS=https://example.com/feed.xml;https://example.com/rss
```

RSS URL 내부에 쉼표가 들어갈 수 있으므로, 여러 피드를 넣을 때는 쉼표가 아니라 세미콜론(`;`) 또는 줄바꿈으로 구분하세요.

권장 기본값:

```text
OPENAI_TEMPERATURE=0.2
MAX_OPENAI_OUTPUT_TOKENS=1500
MAX_ARTICLE_CHARS=2500
MAX_SUMMARY_INPUT_CHARS=1200
MIN_CLEAN_ARTICLE_CHARS=400
MAX_RELATED_LINKS=3
MAX_POSTS_PER_RUN=1
MANUAL_PROMPT_MODE=true
REQUIRE_POST_SUCCESS=true
DRY_RUN=true
PUBLISH_STATUS=draft
```

`OPENAI_MODEL`은 워크플로우에서 `gpt-4o-mini`로 고정되어 있습니다.

## 수동 프롬프트 모드

OpenAI API 결제 전에는 아래 설정을 권장합니다.

```text
MANUAL_PROMPT_MODE=true
DRY_RUN=true
```

이 모드에서는 RSS 수집, 중복 차단, 빈 기사 필터링, 본문 압축까지만 자동으로 수행하고, `output/` 폴더에 ChatGPT 웹에 붙여넣을 `.md` 프롬프트 파일을 생성합니다. GitHub Actions 실행 후 `generated-output` artifact를 내려받아 프롬프트를 복사한 뒤, 유료 ChatGPT 화면에 붙여넣으면 됩니다.

## DRY_RUN 테스트

처음에는 GitHub Variables에서 아래처럼 설정하세요.

```text
DRY_RUN=true
PUBLISH_STATUS=draft
```

GitHub `Actions` 탭에서 `Blogger Auto Post` -> `Run workflow`를 실행하면 Blogger 업로드는 하지 않고 `output/` 폴더에 완성 HTML과 AI JSON 결과를 저장합니다. 워크플로우의 `dry-run-output` artifact에서 결과물을 확인할 수 있습니다.

결과가 괜찮으면:

```text
DRY_RUN=false
PUBLISH_STATUS=draft
```

로 바꿔 Blogger 임시저장 업로드를 확인하고, 마지막에 자동 발행을 원하면:

```text
PUBLISH_STATUS=publish
```

로 변경하세요.

## 자동 실행

워크플로우는 월-금 한국 시간 오전 7시 전후 발행을 목표로 합니다.

```yaml
cron: "50 21 * * 0-4"
```

GitHub Actions cron은 UTC 기준입니다. 위 설정은 한국 시간 월-금 06:50에 해당하며, GitHub 스케줄 지연을 고려한 값입니다.

## 실행 추적

매 실행 후 `run_summary.json`에 아래 메트릭이 누적됩니다.

```text
처리된 총 기사 수
중복 스킵 수
빈 기사 스킵 수
OpenAI API 호출 횟수
입력/출력 글자 수
입력/출력 토큰 수
업로드 성공 수
실패 기사 수
오류 메시지
```

`history.json`에는 발행한 글의 GUID, URL, 제목, Blogger URL, 카테고리, 감지된 티커가 저장됩니다. 이 파일은 다음 실행에서 중복 차단과 "함께 보면 좋은 글" 내부 링크 빌딩에 함께 사용됩니다.

`REQUIRE_POST_SUCCESS=true`이면 `DRY_RUN=false`인 실제 업로드 모드에서 Blogger 글이 0건 생성될 경우 GitHub Actions를 실패 처리합니다. 따라서 OpenAI 쿼터 부족처럼 실제 발행이 없었던 실행이 초록 체크로 보이지 않습니다.

## 비용 메모

OpenAI 공식 가격표 기준 `gpt-4o-mini`는 텍스트 입력 $0.15 / 1M tokens, 출력 $0.60 / 1M tokens로 매우 저렴한 편입니다. 이 봇은 평일 하루 1건, 기사당 1회 호출, 입력 본문 1,200자 제한, 빈 기사 사전 스킵 구조라 월 20~23회 실행 기준 보통 월 100원 이하 수준으로 운영될 가능성이 큽니다. 실제 비용은 RSS 본문 길이와 출력 길이에 따라 달라지므로 OpenAI Usage 화면에서 확인하세요.

가격은 바뀔 수 있으니 운영 전 공식 가격표를 확인하세요: https://platform.openai.com/docs/pricing
