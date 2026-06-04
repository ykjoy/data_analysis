# =====================================================================
# 공공 민원 답변 자동화 (Multi-Agent Demo)
# 비전문가를 위한 멀티 에이전트 입문 강의 실습 코드
# Python · Streamlit · Google Gemini API (OpenAI 호환 엔드포인트)
# =====================================================================

# ===== Block 1. import & 페이지 설정 =====
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="공공민원 자동화", page_icon="🏛️", layout="wide")
st.title("🏛️ 공공--- 민원 답변 자동화 (Multi-Agent Demo)")

# ===== Block 2. Gemini 클라이언트 생성 (OpenAI 호환 엔드포인트 사용) =====
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]   # Streamlit Secrets에서 키 불러오기
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL = "gemini-2.5-flash"   # 무료 티어로 충분, 빠르고 정확


# ===== Block 3. Agent 1 — 분류 에이전트 =====
def classify_minwon(minwon_text: str) -> str:
    """민원 텍스트를 5개 카테고리 중 하나로 분류"""
    system_prompt = """당신은 공공기관 민원 분류 전문가입니다.
입력된 민원을 반드시 아래 5개 카테고리 중 하나로만 분류하세요.
- 도로/시설, 환경/위생, 복지/보건, 세무/재정, 기타
출력 형식: 카테고리명 한 단어만 (예: 도로/시설)"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": minwon_text}
        ],
        temperature=0.1   # 분류는 일관성이 중요 → 낮게 설정
    )
    return response.choices[0].message.content.strip()


# ===== Block 4. Agent 2 — 답변 작성 에이전트 =====
def write_answer(minwon_text: str, category: str) -> str:
    """분류된 카테고리에 맞춰 답변 초안 작성"""
    system_prompt = f"""당신은 공공기관 민원 답변 작성 담당자입니다.
민원 카테고리: {category}
다음 원칙을 지켜 답변하세요:
- 공손하고 명확한 어투 ("~드립니다", "~예정입니다")
- 처리 절차 안내 포함
- 200자 이내, 마크다운 사용 금지
- 개인정보·법적 단정 표현 금지"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": minwon_text}
        ],
        temperature=0.5   # 약간의 자연스러움 허용
    )
    return response.choices[0].message.content.strip()


# ===== Block 5. Agent 3 — 검토 에이전트 + 메인 흐름 =====
def review_answer(answer: str) -> str:
    """답변 초안을 검토하고 최종본 출력"""
    system_prompt = """당신은 민원 답변 품질 검토자입니다.
- 어투가 공손한지, 모호한 표현이 없는지 검토
- 부적절한 표현은 수정해서 '최종 답변'만 출력
- 추가 설명·이모지·머리말 없이 답변 본문만 출력"""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": answer}],
        temperature=0.2)
    return r.choices[0].message.content.strip()


# ===== 메인 흐름 =====
minwon = st.text_area("💬 민원을 입력하세요", height=120)
if st.button("🚀 답변 생성", type="primary") and minwon:
    with st.spinner("Agent 1: 분류 중..."):
        category = classify_minwon(minwon)
    with st.expander(f"🏷️ Agent 1 — 분류: **{category}**", expanded=True):
        st.write(category)
    with st.spinner("Agent 2: 답변 작성 중..."):
        draft = write_answer(minwon, category)
    with st.expander("✍️ Agent 2 — 초안", expanded=True):
        st.write(draft)
    with st.spinner("Agent 3: 검토 중..."):
        final = review_answer(draft)
    st.success("✅ 최종 답변")
    st.write(final)
