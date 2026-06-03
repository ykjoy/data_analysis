# 공공 민원 답변 자동화 (Multi-Agent Demo)

강의 "비전문가를 위한 멀티 에이전트 입문" 실습 코드입니다.
시민 민원 → 분류 → 답변 작성 → 검토, 3개의 AI 에이전트가 협업합니다.

## 파일
- `app.py` — Streamlit 앱 (에이전트 3종 + 메인 흐름)
- `requirements.txt` — 필요한 라이브러리 (streamlit, openai)
- `secrets.toml.example` — API 키 설정 예시

## 로컬 실행
1. `pip install -r requirements.txt`
2. `.streamlit/secrets.toml` 파일을 만들고 `secrets.toml.example` 내용을 채워넣기
   (Google AI Studio: https://aistudio.google.com 에서 무료 발급)
3. `streamlit run app.py`

## Streamlit Cloud 배포
1. `app.py`, `requirements.txt` 를 Public GitHub 저장소에 업로드
2. https://share.streamlit.io → New app → 저장소/브랜치/`app.py` 선택
3. Advanced settings → Secrets 에 `GEMINI_API_KEY = "발급받은키"` 입력
4. Deploy → 1~2분 후 공개 URL 발급

## 주의 (Guardrails)
- API 키는 코드에 직접 쓰지 말고 반드시 Secrets에 보관
- 실습은 실제 시민 민원이 아닌 가상 예시로만
- AI 답변은 초안일 뿐 — 최종 검토·확정은 사람이 (Human-in-the-Loop)
