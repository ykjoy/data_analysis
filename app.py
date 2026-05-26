# -*- coding: utf-8 -*-
"""
================================================================================
 제조 데이터 분석 실습 앱  (Teaching Edition)
================================================================================

이 앱은 비-IT 실무자를 위한 데이터 분석 입문 교재입니다.
"클릭만" 으로 끝나는 도구가 아니라, 코드를 같이 읽으며 원리를 익히세요.

설계 원칙
---------
  1. 본인 데이터로 직접 실습 — 샘플 데이터 자동 생성 없음, 파일 업로드 필수
  2. 단계마다 "왜?" 를 설명 — 무엇을 하는지가 아니라, 왜 그렇게 하는지
  3. 모델 비교 — RF · XGBoost · MLP 등을 같은 데이터로 한 번에 비교

준비할 데이터
-------------
  • 분류    : 마지막 컬럼이 정답(불량/정상 등) 인 CSV
              예) UCI SECOM (Kaggle), 자사 품질 검사 데이터
  • 회귀    : 마지막 컬럼이 수치 값(수율·수명·원가 등) 인 CSV
  • 시계열  : date(날짜) + value(값) 두 컬럼만 있는 CSV
              예) 일별/월별 수요량, 발전량, 생산량
  • HF      : 분석할 이미지 파일 또는 텍스트

실행
----
  $ pip install -r requirements.txt
  $ streamlit run app.py
================================================================================
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# 한글 폰트 설정
# ----------------------------------------------------------------------------
# matplotlib 차트에 한글이 깨지지 않게 OS별 한글 폰트를 자동 설정합니다.
# (Windows: 맑은 고딕 / Mac: AppleGothic / Linux: NanumGothic 또는 Noto)
# ============================================================================
def setup_korean_font():
    import matplotlib.font_manager as fm
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "Noto Sans CJK KR", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False     # 마이너스 부호 깨짐 방지

setup_korean_font()


# ============================================================================
# Streamlit 페이지 설정 + 커스텀 CSS
# ============================================================================
st.set_page_config(
    page_title="제조 데이터 분석 실습",
    page_icon="🏭",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        background: #1E3A5F;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.6rem; }
    .main-header p  { color: #D8E2EA; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #1E3A5F; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 공통 헬퍼 함수
# ----------------------------------------------------------------------------
# CSV 인코딩 자동 감지 — 한국 공공데이터는 보통 CP949, Kaggle 은 UTF-8
# 매번 사용자에게 인코딩을 물어보지 않고 자동으로 시도합니다.
# ============================================================================
def smart_read_csv(file):
    """다양한 한글 인코딩을 자동으로 시도하면서 CSV 를 읽음."""
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception:
            continue
    # 어떤 인코딩으로도 안 되면, 에러를 무시하고 읽기
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8", errors="ignore")


def require_upload(message):
    """파일 업로드가 안 되었을 때 안내 메시지를 보여주고 실행을 멈춤."""
    st.info(message)
    st.stop()


# ============================================================================
# 사이드바 — 메뉴 선택
# ============================================================================
st.sidebar.title("🏭 제조 데이터 분석")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "🏠 홈",
        "① 분류 · 불량분석",
        "② 회귀 · 수치예측",
        "③ 시계열 · 수요예측",
        "④ Hugging Face 데모",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("💡 본인 데이터(.csv / 이미지) 를 업로드해서 실습하세요.")
st.sidebar.caption("📚 코드 주석을 함께 읽어보세요 — 원리 이해에 도움됩니다.")


# ============================================================================
# 홈 페이지
# ============================================================================
if menu == "🏠 홈":
    st.markdown("""
    <div class="main-header">
        <h1>제조 현장을 위한 데이터 분석 입문</h1>
        <p>Hands-on Workshop · Random Forest · XGBoost · Prophet · Deep Learning · Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👋 환영합니다")
    st.write(
        "이 앱은 IT 비전공 **제조 현장 실무자** 를 위한 데이터 분석 학습 도구입니다. "
        "좌측 메뉴를 선택해 본인 데이터로 직접 실습해 보세요."
    )

    st.info(
        "📝 **이 앱은 샘플 데이터를 자동 생성하지 않습니다.** "
        "각 메뉴에서 본인의 CSV 파일을 업로드해 실습하세요. "
        "데이터가 없다면 Kaggle 의 *UCI SECOM* (분류) "
        "또는 data.go.kr 의 *발전설비 운전현황* 같은 공공 데이터를 받아오세요."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ① 분류·불량분석")
        st.write("**3 가지 분류 모델 동시 비교**")
        st.write("- Random Forest (의사결정 트리 다수결)")
        st.write("- XGBoost (Gradient Boosting — 거의 항상 1등)")
        st.write("- MLP (간단한 딥러닝)")
        st.caption("→ Feature Importance · Confusion Matrix · Accuracy/Precision/Recall/F1")

        st.markdown("#### ② 회귀·수치예측")
        st.write("**수율 · 수명 · 원가 등 수치 예측**")
        st.write("- Linear Regression (가장 기본)")
        st.write("- Ridge (정규화 — 과적합 억제)")
        st.write("- XGBoost Regressor (비선형 패턴 OK)")
        st.caption("→ MAE · RMSE · R²")

    with col2:
        st.markdown("#### ③ 시계열·수요예측")
        st.write("**과거 → 미래 예측 + 분해 분석**")
        st.write("- Prophet (Meta — 추세·계절성·휴일 분해)")
        st.write("- LSTM (딥러닝 — 비선형 패턴)")
        st.caption("→ 예측 차트 · Trend / Weekly / Yearly 분해 · MAE/RMSE/MAPE")

        st.markdown("#### ④ Hugging Face")
        st.write("**사전학습 모델을 한 줄로 사용**")
        st.write("- 텍스트 감성 분석 (DistilBERT)")
        st.write("- 이미지 분류 (ViT)")
        st.caption("→ pipeline() 한 줄로 SOTA 모델 즉시 활용")

    st.markdown("---")
    st.markdown("### 📋 학습 흐름")
    st.write("""
    1. **데이터 준비** — Kaggle · data.go.kr 에서 다운로드 (강의 슬라이드 참고)
    2. **파일 업로드** — 좌측 메뉴 선택 후 CSV 업로드
    3. **컬럼 지정** — 어느 컬럼이 정답(예측 대상) 인지 선택
    4. **모델 학습** — [학습 시작] 버튼
    5. **결과 해석** — 지표만 보지 말고 차트 + Feature Importance 도 함께
    6. **코드 읽기** — `app.py` 의 주석을 함께 읽으면 원리 이해 ↑
    """)


# ============================================================================
# ① 분류 · 불량분석
# ----------------------------------------------------------------------------
# 핵심 흐름:
#   1. 파일 업로드 + 컬럼 검토
#   2. 타겟(정답) 컬럼 지정 → X (입력), y (정답) 분리
#   3. 결측치 처리 + 학습/테스트 분할
#   4. 3 가지 모델 학습 (RF / XGBoost / MLP)
#   5. 같은 테스트셋에서 평가 → 성능 비교 + Feature Importance + Confusion Matrix
# ============================================================================
elif menu == "① 분류 · 불량분석":
    st.markdown("""
    <div class="main-header">
        <h1>① 분류 · 불량분석</h1>
        <p>Random Forest · XGBoost · MLP — 같은 데이터, 3 가지 모델 동시 비교</p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # STEP 1: 데이터 업로드
    # ------------------------------------------------------------
    # 분류 데이터는 행=샘플, 열=공정변수(피처) + 마지막에 정답(타겟) 컬럼.
    # 예) UCI SECOM: 1,567행 × 590개 센서 + Pass/Fail (-1/1) 1개 컬럼
    # ------------------------------------------------------------
    st.markdown("### 1️⃣ 데이터 업로드")
    st.caption("CSV 형식. 각 행이 하나의 샘플(제품·로트), 마지막 컬럼에 정답(Pass/Fail 등) 권장.")
    upload = st.file_uploader("분류용 CSV 파일", type=["csv"], key="cls_up")
    if upload is None:
        require_upload("⬆️ 분류 분석을 시작하려면 CSV 파일을 업로드하세요.")

    df = smart_read_csv(upload)
    st.success(f"✅ 업로드 완료: **{df.shape[0]:,}행 × {df.shape[1]:,}컬럼**")

    with st.expander("📋 데이터 미리보기 (상위 10행)"):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"수치형 컬럼: {df.select_dtypes(include=[np.number]).shape[1]}개 / "
                   f"문자형 컬럼: {df.select_dtypes(exclude=[np.number]).shape[1]}개")

    # ------------------------------------------------------------
    # STEP 2: 타겟 컬럼 지정
    # ------------------------------------------------------------
    # 분류 = 정답이 카테고리(Pass/Fail, 0/1, A/B/C 등) 일 때.
    # 만약 정답이 연속 숫자라면 "② 회귀" 메뉴를 사용하세요.
    # ------------------------------------------------------------
    st.markdown("### 2️⃣ 타겟(정답) 컬럼 선택")
    target_col = st.selectbox(
        "어떤 컬럼이 '정답' 인가요? (불량/정상, Pass/Fail 등)",
        df.columns.tolist(),
        index=len(df.columns) - 1,    # 보통 마지막 컬럼에 정답이 있음
        help="이 컬럼을 예측하는 게 목표입니다. 나머지 컬럼은 모두 X(입력) 로 사용됩니다.",
    )

    # 모델 선택
    st.markdown("### 3️⃣ 학습할 모델 선택")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        use_rf = st.checkbox("🌳 Random Forest", value=True,
                             help="여러 결정 트리의 다수결. 튜닝 없이도 무난. 해석 쉬움.")
    with col_b:
        use_xgb = st.checkbox("⚡ XGBoost", value=True,
                              help="Gradient Boosting. 실무에서 거의 항상 최고 성능.")
    with col_c:
        use_mlp = st.checkbox("🧠 MLP (딥러닝)", value=False,
                              help="간단한 신경망. 데이터가 많을 때 강함.")

    n_estimators = st.slider("트리/부스팅 반복 수 (n_estimators)", 50, 500, 200, step=50,
                             help="많을수록 안정적이지만 학습 시간 증가. 보통 100~500.")

    # ------------------------------------------------------------
    # STEP 3: 학습 시작
    # ------------------------------------------------------------
    if st.button("🚀 학습 시작", type="primary"):
        if not (use_rf or use_xgb or use_mlp):
            st.error("⚠️ 최소 1 개 이상의 모델을 선택하세요.")
            st.stop()

        with st.spinner("학습 중... (수초 ~ 1분 소요)"):
            # ----------------------------------------------------------------
            # ① 데이터 분리: X (입력 피처) vs  y (정답)
            # ----------------------------------------------------------------
            # 타겟 컬럼만 떼어내고, 나머지는 모두 입력으로 사용합니다.
            # ----------------------------------------------------------------
            y_raw = df[target_col]
            X = df.drop(columns=[target_col])

            # ----------------------------------------------------------------
            # ② 수치형 변수만 사용 + 결측치 채우기
            # ----------------------------------------------------------------
            # 머신러닝 모델은 수치만 다룰 수 있어, 문자형 컬럼은 일단 제외.
            # (실무에서는 One-Hot Encoding 으로 변환하지만, 입문 단계에서는 단순화.)
            # 결측치는 중앙값(median) 으로 채움 — 평균보다 이상치에 강함.
            # ----------------------------------------------------------------
            X = X.select_dtypes(include=[np.number])
            if X.shape[1] == 0:
                st.error("⚠️ 수치형 피처가 없습니다. CSV 의 컬럼 형식을 확인하세요.")
                st.stop()
            X = X.fillna(X.median())

            # ----------------------------------------------------------------
            # ③ 타겟 인코딩 — 문자 라벨을 숫자로
            # ----------------------------------------------------------------
            # "Pass"/"Fail" 같은 문자 라벨은 모델이 못 다룹니다.
            # LabelEncoder 로 자동 변환: "Pass" → 0, "Fail" → 1 등
            # ----------------------------------------------------------------
            from sklearn.preprocessing import LabelEncoder
            if y_raw.dtype == object or y_raw.dtype.name == "category":
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
                class_names = list(le.classes_)
            else:
                y = y_raw.values
                class_names = [str(c) for c in sorted(np.unique(y))]

            if len(np.unique(y)) < 2:
                st.error("⚠️ 타겟에 클래스가 2개 이상 있어야 합니다. (예: 정상/불량)")
                st.stop()

            # ----------------------------------------------------------------
            # ④ 학습/테스트 분리 (80% / 20%)
            # ----------------------------------------------------------------
            # 핵심 — '공부한 문제로 시험 보면 안 된다'.
            # test_size=0.2  : 20% 를 시험용으로 따로 떼어둠
            # random_state=42: 매번 같은 분할 (재현 가능성)
            # stratify=y     : 클래스 비율을 train/test 양쪽에서 동일하게 유지
            #                  (불량 6.6% 데이터처럼 불균형일 때 중요)
            # ----------------------------------------------------------------
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            # ----------------------------------------------------------------
            # ⑤ 모델 학습 — RF / XGBoost / MLP
            # ----------------------------------------------------------------
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                confusion_matrix, classification_report,
            )

            results = {}        # 모델 이름 → 평가 지표 dict
            predictions = {}    # 모델 이름 → 예측값
            models = {}         # 모델 이름 → 학습된 모델 객체

            # 평가 지표 계산 함수 (중복 코드 방지)
            def evaluate(y_true, y_pred, num_classes):
                avg = "binary" if num_classes == 2 else "weighted"
                return {
                    "Accuracy":  accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
                    "Recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
                    "F1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
                }

            # ---------- Random Forest ----------
            if use_rf:
                # ─────────────────────────────────────────────────────
                # Random Forest = 여러 결정 트리(decision tree) 의 다수결
                #
                # 주요 하이퍼파라미터:
                #   • n_estimators  : 트리 개수. 많을수록 안정적
                #   • max_depth=None: 각 트리를 끝까지 분기 (가지치기 X)
                #                     None 은 과적합 위험 있지만 RF 는 다수결로 완화됨
                #   • class_weight  : 'balanced' → 클래스 불균형 자동 보정
                #                     (불량 6.6% 같은 데이터에서 필수)
                #   • n_jobs=-1     : CPU 모두 사용 (병렬 학습 → 빠름)
                # ─────────────────────────────────────────────────────
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                )
                rf.fit(X_train, y_train)              # 학습 — 한 줄!
                pred = rf.predict(X_test)             # 예측
                models["Random Forest"] = rf
                predictions["Random Forest"] = pred
                results["Random Forest"] = evaluate(y_test, pred, len(class_names))

            # ---------- XGBoost ----------
            if use_xgb:
                # ─────────────────────────────────────────────────────
                # XGBoost = eXtreme Gradient Boosting
                #
                # RF 와의 차이:
                #   • RF      : 트리들을 '병렬' 로 만들어 다수결
                #   • XGBoost : 트리들을 '순차' 로 만들면서 이전 트리의 오차를 보완
                #
                # 실무 1위 모델 — Kaggle 표 데이터 대회 우승 다수.
                # 단점: 튜닝 파라미터가 많고, 과적합 가능성 약간 높음
                #
                # 주요 하이퍼파라미터:
                #   • n_estimators  : 부스팅 반복 수
                #   • learning_rate : 각 트리의 영향력. 작을수록 안정적, 느림
                #   • max_depth     : 트리 깊이. 6 정도가 표준
                # ─────────────────────────────────────────────────────
                try:
                    from xgboost import XGBClassifier
                    # 이진 분류 vs 다중 분류 자동 처리
                    objective = "binary:logistic" if len(class_names) == 2 else "multi:softprob"
                    xgb = XGBClassifier(
                        n_estimators=n_estimators,
                        learning_rate=0.1,
                        max_depth=6,
                        objective=objective,
                        random_state=42,
                        n_jobs=-1,
                        eval_metric="logloss",
                    )
                    xgb.fit(X_train, y_train)
                    pred = xgb.predict(X_test)
                    models["XGBoost"] = xgb
                    predictions["XGBoost"] = pred
                    results["XGBoost"] = evaluate(y_test, pred, len(class_names))
                except ImportError:
                    st.warning("⚠️ xgboost 가 설치되지 않았습니다.  pip install xgboost")

            # ---------- MLP (딥러닝) ----------
            if use_mlp:
                # ─────────────────────────────────────────────────────
                # MLP = Multi-Layer Perceptron (다층 퍼셉트론)
                # 가장 기본적인 신경망. 표 형태 데이터에 사용.
                #
                # 신경망 학습 전 핵심: 입력 스케일링
                #   - 센서마다 단위/범위 다름 (예: 온도 0~200, 압력 0~5)
                #   - 큰 값을 가진 센서가 더 큰 영향을 주는 편향 발생
                #   - StandardScaler 로 평균 0, 표준편차 1 로 정규화
                #   - (RF/XGBoost 는 스케일링 불필요 — 트리는 크기 비교만 함)
                #
                # 구조: hidden_layer_sizes=(64, 32)
                #   = 은닉층 2 개 (64 노드 → 32 노드)
                # ─────────────────────────────────────────────────────
                from sklearn.neural_network import MLPClassifier
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)    # train 으로 fit
                X_test_s = scaler.transform(X_test)          # test 는 transform 만 (정보 누출 방지)
                mlp = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,    # 과적합 자동 방지 — 검증 점수가 안 오르면 중단
                )
                mlp.fit(X_train_s, y_train)
                pred = mlp.predict(X_test_s)
                models["MLP"] = mlp
                predictions["MLP"] = pred
                results["MLP"] = evaluate(y_test, pred, len(class_names))

        st.success("🎉 학습 완료!")

        # ----------------------------------------------------------------
        # ⑥ 결과 비교 — 평가 지표 표
        # ----------------------------------------------------------------
        st.markdown("### 4️⃣ 모델 비교 — 어느 모델이 최고?")
        compare_df = pd.DataFrame(results).T.applymap(lambda x: f"{x*100:.1f}%")
        compare_df.index.name = "모델"
        st.dataframe(compare_df, use_container_width=True)

        st.caption(
            "💡 **무엇을 봐야 하나?**  Accuracy 만 보면 안 됩니다. "
            "불량 6% 데이터라면 모두 '정상' 으로 찍어도 Accuracy 94% — 의미 없음. "
            "**불량 케이스에 대한 Recall(재현율)** 이 더 중요합니다."
        )
        st.markdown("---")

        # ----------------------------------------------------------------
        # ⑦ Feature Importance — '어떤 변수가 가장 중요한가?'
        # ----------------------------------------------------------------
        # RF 와 XGBoost 는 학습 후 .feature_importances_ 를 제공합니다.
        # 막대가 길수록 그 센서/변수가 불량 예측에 더 큰 영향.
        # → 공정 개선 우선순위 의사결정에 직접 활용 가능
        # ----------------------------------------------------------------
        st.markdown("### 5️⃣ Feature Importance — 어떤 변수가 중요한가?")
        fi_models = [m for m in ["Random Forest", "XGBoost"] if m in models]
        if fi_models:
            cols_fi = st.columns(len(fi_models))
            for i, name in enumerate(fi_models):
                fi = pd.Series(models[name].feature_importances_, index=X.columns)
                fi = fi.sort_values(ascending=True).tail(10)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(fi.index, fi.values, color="#F47C3C")
                ax.set_xlabel("Importance")
                ax.set_title(f"{name} — Top 10")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                cols_fi[i].pyplot(fig)
            st.caption(
                "💡 RF 와 XGBoost 의 Top 변수가 비슷하다면 → 결과 신뢰 ↑.  "
                "다르다면 → 데이터 더 깊이 보세요 (Correlation, EDA)."
            )
        st.markdown("---")

        # ----------------------------------------------------------------
        # ⑧ Confusion Matrix — 어디서 틀렸는지 확인
        # ----------------------------------------------------------------
        # 대각선 = 맞춘 것. 비대각선 = 틀린 것.
        # 제조에서는 특히 FN(불량을 정상으로 본 것) 이 가장 치명적입니다.
        # → 불량품이 출하 → 고객 클레임 → 회사 손실
        # ----------------------------------------------------------------
        st.markdown("### 6️⃣ Confusion Matrix — 어디서 틀렸나?")
        cols_cm = st.columns(len(predictions))
        for i, (name, pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
            ax.set_xlabel("예측"); ax.set_ylabel("실제")
            for r in range(len(class_names)):
                for c in range(len(class_names)):
                    txtcolor = "white" if cm[r, c] > cm.max() / 2 else "black"
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                            color=txtcolor, fontsize=14, fontweight="bold")
            ax.set_title(f"{name}")
            fig.tight_layout()
            cols_cm[i].pyplot(fig)

        # 상세 분류 리포트
        with st.expander("📊 상세 분류 리포트 (클래스별 Precision · Recall · F1)"):
            for name, pred in predictions.items():
                st.markdown(f"**{name}**")
                st.code(classification_report(y_test, pred, target_names=class_names, zero_division=0))


# ============================================================================
# ② 회귀 · 수치예측
# ----------------------------------------------------------------------------
# 분류와의 차이: 정답이 카테고리가 아니라 '연속된 숫자'
#   • 분류 예) 불량(1)/정상(0)
#   • 회귀 예) 수율 92.5%, 수명 1,847 시간, 원가 $3.47
#
# 핵심 흐름은 분류와 거의 같습니다. 모델만 *Classifier → *Regressor 로 변경.
# 평가 지표만 다름: Accuracy 대신 MAE / RMSE / R²
# ============================================================================
elif menu == "② 회귀 · 수치예측":
    st.markdown("""
    <div class="main-header">
        <h1>② 회귀 · 수치예측</h1>
        <p>Linear · Ridge · XGBoost Regressor — 수율 · 수명 · 원가 등 수치 예측</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1️⃣ 데이터 업로드")
    st.caption("CSV 형식. 마지막 컬럼이 '예측할 숫자' (수율·수명·온도 등) 권장.")
    upload = st.file_uploader("회귀용 CSV 파일", type=["csv"], key="reg_up")
    if upload is None:
        require_upload("⬆️ 회귀 분석을 시작하려면 CSV 파일을 업로드하세요.")

    df = smart_read_csv(upload)
    st.success(f"✅ 업로드 완료: **{df.shape[0]:,}행 × {df.shape[1]:,}컬럼**")
    with st.expander("📋 데이터 미리보기"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 2️⃣ 타겟(예측 대상) 컬럼 선택")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("⚠️ 수치형 컬럼이 없습니다. 회귀는 숫자 예측용입니다.")
        st.stop()
    target_col = st.selectbox(
        "예측할 숫자 컬럼", numeric_cols,
        index=len(numeric_cols) - 1,
    )

    st.markdown("### 3️⃣ 학습할 모델")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        use_lin = st.checkbox("📐 Linear Regression", value=True,
                              help="가장 단순. 직선/평면으로 데이터에 맞춤. 베이스라인.")
    with col_b:
        use_ridge = st.checkbox("🛡️ Ridge", value=True,
                                help="Linear + 정규화. 과적합 억제. 변수가 많을 때 추천.")
    with col_c:
        use_xgbr = st.checkbox("⚡ XGBoost Regressor", value=True,
                               help="비선형 패턴까지 학습. 실무 1위 회귀 모델.")

    if st.button("🚀 학습 시작", type="primary"):
        if not (use_lin or use_ridge or use_xgbr):
            st.error("⚠️ 최소 1 개 모델을 선택하세요."); st.stop()

        with st.spinner("학습 중..."):
            # ----------------------------------------------------------------
            # ① X, y 분리 + 결측치 처리 + train/test 분할
            # ----------------------------------------------------------------
            # 분류와 차이: stratify 안 함 (연속값은 stratify 불가).
            # ----------------------------------------------------------------
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            y = df[target_col].values
            X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
            X = X.fillna(X.median())

            mask = ~np.isnan(y)
            X, y = X[mask], y[mask]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
            )

            results = {}
            preds = {}
            models = {}

            # ─────────────────────────────────────────────────────
            # Linear Regression — y = w₁x₁ + w₂x₂ + … + b
            # 가장 단순한 모델. 회귀의 베이스라인으로 항상 먼저 돌려봅니다.
            # 장점: 해석 쉬움 (각 변수의 계수 = 영향력 방향+크기)
            # 단점: 비선형 관계 못 잡음
            # ─────────────────────────────────────────────────────
            if use_lin:
                from sklearn.linear_model import LinearRegression
                lin = LinearRegression()
                lin.fit(X_train, y_train)
                p = lin.predict(X_test)
                models["Linear"] = lin; preds["Linear"] = p
                results["Linear"] = (mean_absolute_error(y_test, p),
                                     np.sqrt(mean_squared_error(y_test, p)),
                                     r2_score(y_test, p))

            # ─────────────────────────────────────────────────────
            # Ridge Regression — Linear + L2 정규화
            # 큰 계수에 페널티를 줘서 과적합 억제.
            # alpha: 정규화 강도. 클수록 계수가 작아짐(단순화).
            # 변수가 많거나 변수끼리 상관관계가 클 때 Linear 보다 안정적.
            # ─────────────────────────────────────────────────────
            if use_ridge:
                from sklearn.linear_model import Ridge
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train, y_train)
                p = ridge.predict(X_test)
                models["Ridge"] = ridge; preds["Ridge"] = p
                results["Ridge"] = (mean_absolute_error(y_test, p),
                                    np.sqrt(mean_squared_error(y_test, p)),
                                    r2_score(y_test, p))

            # ─────────────────────────────────────────────────────
            # XGBoost Regressor — Gradient Boosting for regression
            # 트리 기반이므로 비선형 관계, 변수 간 상호작용 자동 학습.
            # 실무에서 회귀도 거의 항상 XGBoost 가 가장 정확.
            # ─────────────────────────────────────────────────────
            if use_xgbr:
                try:
                    from xgboost import XGBRegressor
                    xgbr = XGBRegressor(
                        n_estimators=200, learning_rate=0.1, max_depth=6,
                        random_state=42, n_jobs=-1,
                    )
                    xgbr.fit(X_train, y_train)
                    p = xgbr.predict(X_test)
                    models["XGBoost"] = xgbr; preds["XGBoost"] = p
                    results["XGBoost"] = (mean_absolute_error(y_test, p),
                                          np.sqrt(mean_squared_error(y_test, p)),
                                          r2_score(y_test, p))
                except ImportError:
                    st.warning("⚠️ xgboost 가 설치되지 않았습니다.")

        st.success("🎉 학습 완료!")

        # ----------------------------------------------------------------
        # 평가 지표 — 회귀 전용
        # ----------------------------------------------------------------
        # MAE : 평균 절대 오차. 단위 그대로 (예: 평균 3.2 어긋남)
        # RMSE: 제곱근 오차. 큰 오차에 더 큰 페널티
        # R²  : 결정계수. 1.0 에 가까울수록 좋음. 0 이면 평균값만큼만 예측.
        #       음수 나오면 모델이 평균보다 못함 = 사용 X
        # ----------------------------------------------------------------
        st.markdown("### 4️⃣ 평가 — 어느 모델이 최고?")
        rows = []
        for name, (mae, rmse, r2) in results.items():
            rows.append({"모델": name,
                         "MAE": f"{mae:.3f}",
                         "RMSE": f"{rmse:.3f}",
                         "R² (결정계수)": f"{r2:.3f}"})
        st.dataframe(pd.DataFrame(rows).set_index("모델"), use_container_width=True)
        st.caption(
            "💡  **R²** : 1.0 에 가까울수록 좋음. 0.7 이상이면 실무 활용 가능 수준. "
            "0.3 미만이면 데이터 부족 또는 변수 추가 필요."
        )

        # ----------------------------------------------------------------
        # 예측 vs 실제 — 산점도
        # ----------------------------------------------------------------
        st.markdown("### 5️⃣ 예측 vs 실제 — 대각선에 가까울수록 좋음")
        cols_sc = st.columns(len(preds))
        for i, (name, p) in enumerate(preds.items()):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, p, alpha=0.5, color="#1E3A5F", s=15)
            lims = [min(y_test.min(), p.min()), max(y_test.max(), p.max())]
            ax.plot(lims, lims, "--", color="#F47C3C", linewidth=1.5, label="완벽한 예측")
            ax.set_xlabel("실제값"); ax.set_ylabel("예측값")
            ax.set_title(f"{name}  (R²={results[name][2]:.3f})")
            ax.legend(loc="upper left", fontsize=9)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            fig.tight_layout()
            cols_sc[i].pyplot(fig)


# ============================================================================
# ③ 시계열 · 수요예측
# ----------------------------------------------------------------------------
# Prophet (Meta) + LSTM (옵션) — 핵심 흐름은 5 단계
#   1. 날짜·값 컬럼 지정
#   2. ds (date), y (value) 형식으로 변환
#   3. Prophet 학습
#   4. 미래 N 일 예측
#   5. 추세·계절성 분해로 결과 해석
# ============================================================================
elif menu == "③ 시계열 · 수요예측":
    st.markdown("""
    <div class="main-header">
        <h1>③ 시계열 · 수요예측</h1>
        <p>Prophet (Meta) · LSTM (딥러닝) — 과거에서 미래로</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 1️⃣ 데이터 업로드")
    st.caption("CSV 형식. 날짜 컬럼 + 값 컬럼이 있어야 합니다. (예: date, value)")
    upload = st.file_uploader("시계열 CSV 파일", type=["csv"], key="ts_up")
    if upload is None:
        require_upload(
            "⬆️ 시계열 예측을 시작하려면 CSV 파일을 업로드하세요.\n\n"
            "필요 형식: 날짜 컬럼 (YYYY-MM-DD) + 값 컬럼 (수요량·생산량 등)"
        )

    df = smart_read_csv(upload)
    st.success(f"✅ 업로드 완료: **{df.shape[0]:,}행 × {df.shape[1]:,}컬럼**")
    with st.expander("📋 데이터 미리보기"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 2️⃣ 날짜 / 값 컬럼 지정")
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("날짜 컬럼", df.columns.tolist(), index=0,
                                help="YYYY-MM-DD, YYYY/MM/DD 등 다양한 형식 자동 인식")
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("⚠️ 수치형 컬럼이 없습니다."); st.stop()
        value_col = st.selectbox("값 컬럼", numeric_cols, index=0)

    horizon = st.slider("예측 기간 (일)", 30, 365, 90, step=30,
                        help="너무 멀리는 신뢰도 떨어짐. 보통 30~180일 권장.")
    also_lstm = st.checkbox("🧠 LSTM (딥러닝) 도 함께 학습 — 1~3분 소요", value=False)

    if st.button("🔮 예측 시작", type="primary"):
        with st.spinner("Prophet 학습 중..."):
            # ----------------------------------------------------------------
            # ① Prophet 입력 형식으로 변환
            # ----------------------------------------------------------------
            # Prophet 은 반드시 다음 두 컬럼 이름을 요구합니다:
            #   • ds : datestamp (날짜)
            #   • y  : 예측할 값
            # ----------------------------------------------------------------
            data = df[[date_col, value_col]].copy()
            data.columns = ["ds", "y"]
            data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
            data = data.dropna().sort_values("ds").reset_index(drop=True)

            if len(data) < 30:
                st.error("⚠️ 시계열 학습에는 최소 30개 이상의 데이터가 필요합니다.")
                st.stop()

            # ----------------------------------------------------------------
            # ② Prophet 모델 생성 & 학습
            # ----------------------------------------------------------------
            # Prophet 가산 모델 :
            #   y(t) = g(t) + s(t) + h(t) + ε(t)
            #          추세    계절성  휴일   잡음
            #
            # 옵션 설명:
            #   • yearly_seasonality=True  : 연간 패턴 (성수기/비수기) 자동 탐지
            #   • weekly_seasonality=True  : 요일별 패턴 (월~일) 자동 탐지
            #   • daily_seasonality=False  : 일내 시간별 패턴 — 일별 데이터엔 불필요
            #
            # Prophet 의 강점:
            #   - 결측치·이상치에 강함
            #   - 휴일 효과를 별도로 모델링
            #   - 학습 빠름 (수초~수십초)
            # ----------------------------------------------------------------
            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            m.fit(data)              # 학습 — 한 줄!

            # ----------------------------------------------------------------
            # ③ 미래 데이터프레임 만들고 예측
            # ----------------------------------------------------------------
            # make_future_dataframe(periods=N) 는
            # 학습 데이터 + N 일치 빈 행을 더한 dataframe 을 반환
            # predict() 가 이 dataframe 의 모든 날짜에 대해 yhat (예측값) 채움
            # ----------------------------------------------------------------
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)

            # ----------------------------------------------------------------
            # ④ 학습 데이터에 대한 평가 (in-sample)
            # ----------------------------------------------------------------
            # 진짜 평가는 cross_validation() 이 필요하지만, 입문 단계에선 단순화.
            # 학습 점수가 너무 좋으면 (MAPE < 1%) 오히려 과적합 의심.
            # ----------------------------------------------------------------
            train_pred = forecast.iloc[:len(data)]["yhat"].values
            train_true = data["y"].values
            mae = mean_absolute_error(train_true, train_pred)
            rmse = np.sqrt(mean_squared_error(train_true, train_pred))
            mape = np.mean(np.abs((train_true - train_pred) /
                                  np.where(train_true == 0, 1, train_true))) * 100

        st.success("🎉 예측 완료!")

        # ----------------------------------------------------------------
        # 평가 지표
        # ----------------------------------------------------------------
        # MAE  : 단위 그대로 (예: 평균 12개 어긋남)
        # RMSE : 큰 오차에 페널티 (가끔 크게 틀리는 게 치명적일 때)
        # MAPE : % 표현 (임원 보고용 — "평균 8% 오차")
        # ----------------------------------------------------------------
        st.markdown("### 3️⃣ 정확도 (학습 데이터 기준)")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")

        # 예측 차트
        st.markdown("### 4️⃣ 예측 결과")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(data["ds"], data["y"], color="#1E3A5F", label="실제값", linewidth=1.2)
        ax.plot(forecast["ds"], forecast["yhat"], color="#F47C3C", label="예측값", linewidth=1.5)
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                        color="#F47C3C", alpha=0.15, label="신뢰구간 (80%)")
        last = data["ds"].max()
        ax.axvline(last, color="gray", linestyle="--", alpha=0.6)
        ax.text(last, ax.get_ylim()[1] * 0.95, "  예측 시작", color="gray", fontsize=9)
        ax.set_xlabel("날짜"); ax.set_ylabel("값")
        ax.set_title(f"{horizon}일 예측")
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

        # 분해 차트 — Prophet 의 가장 큰 강점
        st.markdown("### 5️⃣ 추세 · 계절성 분해")
        st.caption("Prophet 의 핵심 강점 — '왜 그렇게 예측했는지' 가 설명됨")
        fig_comp = m.plot_components(forecast)
        st.pyplot(fig_comp)

        with st.expander("📋 예측값 상세 (마지막 30일)"):
            show = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30).copy()
            show.columns = ["날짜", "예측값", "하한", "상한"]
            show["날짜"] = show["날짜"].dt.strftime("%Y-%m-%d")
            for c in ["예측값", "하한", "상한"]:
                show[c] = show[c].round(2)
            st.dataframe(show, use_container_width=True)

        # ----------------------------------------------------------------
        # LSTM (선택 — 딥러닝 비교)
        # ----------------------------------------------------------------
        # LSTM = Long Short-Term Memory
        # RNN 의 일종. 과거의 어떤 정보를 기억할지/잊을지 학습.
        # 시계열·자연어처럼 '순서가 중요한 데이터' 에 강함.
        # ----------------------------------------------------------------
        if also_lstm:
            st.markdown("---")
            st.markdown("### 🧠 LSTM (딥러닝) — Prophet 과 비교")
            with st.spinner("LSTM 학습 중... (1~3분)"):
                try:
                    import torch
                    import torch.nn as nn
                    from sklearn.preprocessing import MinMaxScaler

                    # ─── 입력 데이터 준비 ───
                    # LSTM 은 [샘플, 시간, 피처] 3D 텐서를 입력으로 받습니다.
                    # SEQ_LEN=30 : 과거 30 일을 보고 다음 1 일을 예측
                    series = data["y"].values.astype(np.float32)
                    scaler = MinMaxScaler()
                    s_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
                    SEQ = 30
                    Xs, ys = [], []
                    for i in range(len(s_scaled) - SEQ):
                        Xs.append(s_scaled[i:i + SEQ])
                        ys.append(s_scaled[i + SEQ])
                    Xs = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
                    ys = torch.tensor(np.array(ys), dtype=torch.float32)

                    # ─── 모델 정의 ───
                    # LSTM 층 1개 (hidden=32) + 출력층 1개
                    # 매우 단순한 구조 — 실무에선 더 깊게 쌓기도 합니다.
                    class TinyLSTM(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.lstm = nn.LSTM(1, 32, batch_first=True)
                            self.fc = nn.Linear(32, 1)
                        def forward(self, x):
                            o, _ = self.lstm(x)
                            return self.fc(o[:, -1, :]).squeeze(-1)

                    # ─── 학습 루프 ───
                    # Adam optimizer + MSE loss, 50 epoch
                    model = TinyLSTM()
                    opt = torch.optim.Adam(model.parameters(), lr=0.01)
                    loss_fn = nn.MSELoss()
                    model.train()
                    for epoch in range(50):
                        opt.zero_grad()
                        pred = model(Xs)
                        loss = loss_fn(pred, ys)
                        loss.backward()
                        opt.step()

                    # ─── 미래 예측 (재귀적) ───
                    # 1일 예측 → 그걸 입력에 추가 → 다음 1일 예측 → 반복
                    model.eval()
                    last_seq = s_scaled[-SEQ:].tolist()
                    preds = []
                    with torch.no_grad():
                        for _ in range(horizon):
                            inp = torch.tensor(last_seq[-SEQ:], dtype=torch.float32).reshape(1, SEQ, 1)
                            p = model(inp).item()
                            preds.append(p)
                            last_seq.append(p)
                    preds_real = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                    future_dates = pd.date_range(data["ds"].max() + pd.Timedelta(days=1),
                                                 periods=horizon, freq="D")

                    fig_lstm, ax_lstm = plt.subplots(figsize=(11, 4))
                    ax_lstm.plot(data["ds"], data["y"], color="#1E3A5F", label="실제값", linewidth=1)
                    ax_lstm.plot(future_dates, preds_real, color="#6C7A89",
                                 label="LSTM 예측", linewidth=1.5)
                    fut_pp = forecast.tail(horizon)
                    ax_lstm.plot(fut_pp["ds"], fut_pp["yhat"], color="#F47C3C",
                                 label="Prophet 예측", linewidth=1.5, linestyle="--")
                    ax_lstm.set_title("LSTM  vs  Prophet — 동일 기간 예측")
                    ax_lstm.legend()
                    ax_lstm.spines["top"].set_visible(False)
                    ax_lstm.spines["right"].set_visible(False)
                    fig_lstm.tight_layout()
                    st.pyplot(fig_lstm)
                    st.caption("💡 LSTM 은 학습이 더 무겁지만 복잡한 비선형 패턴을 더 잘 잡을 수 있습니다.")
                except ImportError:
                    st.warning("⚠️ PyTorch 가 설치되지 않았습니다. requirements.txt 의 torch 설치 필요.")


# ============================================================================
# ④ Hugging Face 데모
# ----------------------------------------------------------------------------
# 사전학습 모델을 한 줄 코드로 사용 — pipeline(작업종류) 만 호출하면 끝.
# 처음 사용 시 모델 자동 다운로드 (1~3분), 이후엔 캐시에서 즉시 로드.
# ============================================================================
elif menu == "④ Hugging Face 데모":
    st.markdown("""
    <div class="main-header">
        <h1>④ Hugging Face — 사전학습 모델</h1>
        <p>전 세계 100만+ 모델을 한 줄 코드로 — 학습 없이 즉시 사용</p>
    </div>
    """, unsafe_allow_html=True)

    demo_type = st.radio(
        "데모 종류",
        ["📝 텍스트 감성 분석", "🖼️ 이미지 분류"],
        horizontal=True,
    )

    # ----------------------------------------------------------------
    # 텍스트 — 감성 분석
    # ----------------------------------------------------------------
    if demo_type == "📝 텍스트 감성 분석":
        st.markdown("""
        **모델**: `distilbert-base-uncased-finetuned-sst-2-english`
        영문 텍스트의 긍정/부정 자동 판별. 고객 클레임·리뷰 분석 응용.
        """)
        st.markdown("---")
        st.info("💡 분석할 영문 텍스트를 직접 입력해 주세요 (한 줄에 하나씩).")
        text_input = st.text_area(
            "분석할 텍스트",
            value="",
            height=140,
            placeholder=(
                "The product quality is excellent.\n"
                "Defect rate has been too high this month.\n"
                "Customer service was very helpful."
            ),
        )

        if st.button("🤖 분석 시작", type="primary"):
            if not text_input.strip():
                st.warning("⚠️ 텍스트를 입력하세요."); st.stop()
            with st.spinner("모델 다운로드 & 분석 중... (첫 실행 1~2분)"):
                try:
                    # ──────────────────────────────────────────
                    # pipeline() 한 줄로 SOTA 모델 로딩
                    # 1) 작업 종류 ('sentiment-analysis')
                    # 2) 모델 이름 (HF Hub 의 모델 ID)
                    # transformers 가 모델 자동 다운로드 + 토크나이저 설정
                    # ──────────────────────────────────────────
                    from transformers import pipeline
                    clf = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                    )
                    # 한 줄로 추론 — 입력 리스트 → 결과 리스트
                    lines = [t.strip() for t in text_input.split("\n") if t.strip()]
                    results = clf(lines)
                    df_res = pd.DataFrame({
                        "문장": lines,
                        "판정": [r["label"] for r in results],
                        "확신도": [f"{r['score']*100:.1f}%" for r in results],
                    })
                    st.success("✅ 분석 완료!")
                    st.dataframe(df_res, use_container_width=True)
                    st.caption("💡 영문 일반 텍스트로 학습된 모델 — 도메인 특화엔 Fine-tuning 필요.")
                except Exception as e:
                    st.error(f"⚠️ 모델 로드 실패: {e}")

    # ----------------------------------------------------------------
    # 이미지 분류
    # ----------------------------------------------------------------
    else:
        st.markdown("""
        **모델**: `google/vit-base-patch16-224` (Vision Transformer)
        업로드한 이미지를 1000 개 카테고리로 자동 분류.
        """)
        st.markdown("---")
        up_img = st.file_uploader("이미지 업로드 (jpg / png)",
                                  type=["jpg", "jpeg", "png"], key="hf_img")
        if up_img is None:
            require_upload("⬆️ 분석할 이미지를 업로드하세요.")

        from PIL import Image
        img = Image.open(up_img).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="입력 이미지", use_column_width=True)
        with col2:
            if st.button("🤖 분류 시작", type="primary"):
                with st.spinner("모델 다운로드 & 분류 중... (첫 실행 1~3분)"):
                    try:
                        # 동일한 패턴 — pipeline 한 줄
                        from transformers import pipeline
                        clf = pipeline(
                            "image-classification",
                            model="google/vit-base-patch16-224",
                        )
                        results = clf(img, top_k=5)
                        df_res = pd.DataFrame({
                            "순위": range(1, len(results) + 1),
                            "분류 결과": [r["label"] for r in results],
                            "확신도": [f"{r['score']*100:.1f}%" for r in results],
                        })
                        st.success("✅ 분류 완료!")
                        st.dataframe(df_res, use_container_width=True)
                        st.caption(
                            "💡 ImageNet (일반 이미지) 으로 학습된 모델. "
                            "제조 결함 검출은 YOLOv8 + 자사 사진 데이터로 Fine-tuning 권장."
                        )
                    except Exception as e:
                        st.error(f"⚠️ 모델 로드 실패: {e}")

        st.markdown("---")
        st.markdown("#### 🎯 실제 현장 적용 — YOLO (제조 결함 검출 표준)")
        st.code("""
# 제품 표면 결함 검출 — YOLOv8 + 자사 사진 데이터
from ultralytics import YOLO

model = YOLO("yolov8n.pt")           # 사전학습 가중치 로드
results = model("product.jpg")        # 한 줄로 추론
results[0].show()                     # bounding box 그려진 결과 표시
""", language="python")
