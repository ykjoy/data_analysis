# -*- coding: utf-8 -*-
"""
================================================================================
 제조 데이터 분석 실습 앱  (Teaching Edition) — Lite
================================================================================
구성 메뉴
  ① 분류 · 불량분석       — Random Forest / XGBoost / MLP 비교
  ② 시계열 · 수요예측     — Prophet / LSTM
  ③ Hugging Face 데모     — 텍스트 감성 분석 / 이미지 분류

홈 화면에서 현재 실행 중인 app.py 소스 코드를 다운로드 받을 수 있습니다.
================================================================================
"""

import os
import io
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# 한글 폰트  — 매 차트 직전에도 호출해서 깨짐 방지
# ============================================================================
import os, urllib.request
import matplotlib.font_manager as fm

FONT_URL = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
FONT_PATH = "/tmp/NanumGothic.ttf"

KOR_FONT = None
def setup_korean_font():
    global KOR_FONT
    if KOR_FONT is None:
        if not os.path.exists(FONT_PATH):
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        fm.fontManager.addfont(FONT_PATH)
        KOR_FONT = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["font.family"] = KOR_FONT
    plt.rcParams["axes.unicode_minus"] = False

setup_korean_font()



# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(page_title="제조 데이터 분석 실습", page_icon="🏭", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: #1E3A5F; padding: 1.2rem 1.5rem; border-radius: 6px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.6rem; }
    .main-header p  { color: #D8E2EA; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #1E3A5F; font-weight: bold; }

    /* Streamlit 기본 빨강(#FF4B4B) → 부드러운 코랄톤으로 톤다운 */
    button[kind="primary"], div[data-testid="stBaseButton-primary"] {
        background-color: #D89478 !important;
        border-color: #D89478 !important;
        color: white !important;
    }
    button[kind="primary"]:hover, div[data-testid="stBaseButton-primary"]:hover {
        background-color: #C77F62 !important;
        border-color: #C77F62 !important;
        color: white !important;
    }
    button[kind="primary"]:active, button[kind="primary"]:focus {
        background-color: #B66B4F !important;
        border-color: #B66B4F !important;
        color: white !important;
        box-shadow: none !important;
    }

    /* 코드 expander 의 좌측 보더 살짝 */
    div[data-testid="stExpander"] details {
        border-left: 3px solid #D89478;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 공통 헬퍼
# ============================================================================
def smart_read_csv(file):
    """인코딩 + 구분자 자동 감지, 컬럼명 모두 str로."""
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception:
            continue
    file.seek(0)
    df = pd.read_csv(file, encoding="utf-8", errors="ignore",
                     sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_upload(message):
    st.info(message)
    st.stop()


def clean_numeric_X(X):
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        X = X.drop(columns=all_nan)
    med = X.median(numeric_only=True).fillna(0)
    X = X.fillna(med)
    return X, all_nan


def feature_selector_ui(df, target_col, key_prefix):
    st.markdown("#### 🔧 피처(입력 변수) 선택")
    candidates = [c for c in df.columns if c != target_col]
    sub = df[candidates]
    miss = sub.isna().mean()
    high_miss = miss[miss > 0.5].index.tolist()
    num_sub = sub.select_dtypes(include=[np.number])
    zero_var = num_sub.columns[num_sub.nunique(dropna=True) <= 1].tolist() if not num_sub.empty else []
    non_numeric = sub.select_dtypes(exclude=[np.number]).columns.tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        drop_m = st.checkbox(f"결측 50%↑ 제외 ({len(high_miss)}개)", True, key=f"{key_prefix}_m")
    with c2:
        drop_v = st.checkbox(f"분산 0 제외 ({len(zero_var)}개)", True, key=f"{key_prefix}_v")
    with c3:
        drop_n = st.checkbox(f"비수치형 제외 ({len(non_numeric)}개)", True, key=f"{key_prefix}_n")

    auto = set()
    if drop_m: auto.update(high_miss)
    if drop_v: auto.update(zero_var)
    if drop_n: auto.update(non_numeric)
    default = [c for c in candidates if c not in auto]

    with st.expander(f"📌 사용할 피처 직접 선택 (현재 {len(default)} / 전체 {len(candidates)}개)"):
        selected = st.multiselect("체크된 컬럼만 사용", candidates, default=default,
                                  key=f"{key_prefix}_sel")
    return selected


# ============================================================================
# 사이드바
# ============================================================================
st.sidebar.title("🏭 제조 데이터 분석")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "🏠 홈",
        "① 분류 · 불량분석",
        "② 시계열 · 수요예측",
        "③ Hugging Face 데모",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("💡 본인 데이터(.csv / 이미지) 를 업로드해서 실습하세요.")


# ============================================================================
# 🏠 홈
# ============================================================================
if menu == "🏠 홈":
    st.markdown("""
    <div class="main-header">
        <h1>제조 현장을 위한 데이터 분석 입문</h1>
        <p>Classification · Time Series · Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("좌측 메뉴에서 실습 항목을 선택하세요.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ① 분류 · 불량분석")
        st.write("Random Forest / XGBoost / MLP 동시 비교")
        st.markdown("#### ② 시계열 · 수요예측")
        st.write("Prophet · LSTM")
    with col2:
        st.markdown("#### ③ Hugging Face")
        st.write("텍스트 감성 분석 · 이미지 분류")


# ============================================================================
# ① 분류 · 불량분석
# ============================================================================
elif menu == "① 분류 · 불량분석":
    st.markdown("""
    <div class="main-header"><h1>① 분류 · 불량분석</h1>
    <p>Random Forest · XGBoost · MLP — 3 가지 모델 동시 비교</p></div>
    """, unsafe_allow_html=True)

    # ── 코드 보기 ───────────────────────────────────────────────────
    with st.expander("💻 이 분석의 핵심 코드 보기", expanded=False):
        st.code("""
# ─ 분류 분석 핵심 흐름 ─
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1) X, y 분리
X = df.drop(columns=[target])  # 입력 변수
y = df[target]                 # 정답(불량/정상)

# 2) 자동 피처 선택 — 변수 너무 많으면 노이즈 ↑, 핵심만 추림
selector = SelectKBest(f_classif, k=30)    # 통계적 관련성 Top 30
X_selected = selector.fit_transform(X, y)

# 3) train/test 분할 (불균형 데이터는 stratify 필수)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# 4) 모델 학습 — class_weight 로 불균형 자동 보정
rf = RandomForestClassifier(class_weight='balanced', n_estimators=200)
rf.fit(X_tr, y_tr)

# XGBoost 는 scale_pos_weight (다수클래스/소수클래스 비율)
xgb = XGBClassifier(scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum())
xgb.fit(X_tr, y_tr)

# 5) 평가 — 불균형 데이터에서는 Recall, F1 이 핵심
""", language="python")

    st.markdown("### 1️⃣ 데이터 업로드")
    upload = st.file_uploader("분류용 CSV", type=["csv"], key="cls_up")
    if upload is None:
        require_upload("⬆️ CSV 를 업로드하세요. (예: UCI-SECOM.csv)")

    df = smart_read_csv(upload)
    st.success(f"✅ {df.shape[0]:,}행 × {df.shape[1]:,}컬럼")
    with st.expander("📋 미리보기"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 2️⃣ 타겟(정답) 컬럼")
    target_col = st.selectbox("정답 컬럼", df.columns.tolist(),
                              index=len(df.columns)-1)
    tgt = df[target_col].value_counts(dropna=False)
    st.caption("**타겟 분포**: " + ", ".join([f"`{k}` → {v:,}" for k, v in tgt.items()]))

    # ── 클래스 불균형 자동 진단 ─────────────────────────────────
    if len(tgt) >= 2:
        minority_ratio = tgt.min() / tgt.sum()
        if minority_ratio < 0.15:
            st.warning(
                f"⚠️ **클래스 불균형 감지** — 소수 클래스 비율 {minority_ratio*100:.1f}%\n\n"
                f"• 모델이 모두 다수 클래스로 찍어도 정확도가 {(1-minority_ratio)*100:.0f}% 가 나옵니다.\n"
                f"• Accuracy 보다 **Recall · F1** 을 봐야 합니다.\n"
                f"• 아래 **④ 불균형 처리** 섹션에서 리샘플링·임계값을 조정해 보세요. "
                f"`class_weight=balanced` 만으로는 부족한 경우가 많습니다."
            )

    selected = feature_selector_ui(df, target_col, "cls")
    if not selected:
        st.error("⚠️ 피처를 1개 이상 선택하세요."); st.stop()

    # ── ★ 자동 피처 선택 (NEW) ─────────────────────────────────
    st.markdown("### 3️⃣ 자동 피처 선택 ✨")
    st.caption(
        "변수가 너무 많으면 노이즈가 많아져 모델 성능이 떨어집니다. "
        "타겟과 통계적으로 관련 높은 Top K 개만 자동으로 골라냅니다."
    )
    n_selected = len(selected)
    use_auto = st.checkbox(
        f"🎯 자동 피처 선택 사용 (현재 후보 {n_selected}개)",
        value=(n_selected > 50),
        help="UCI-SECOM 처럼 변수 400+ 인 경우 강력 권장. 보통 20~50개로 줄임."
    )
    top_k, sel_method = None, None
    if use_auto:
        c1, c2 = st.columns([2, 1])
        with c1:
            max_k = max(5, min(n_selected, 200))
            default_k = min(30, max_k)
            top_k = st.slider("선택할 피처 수 (K)", 5, max_k, default_k, 5,
                              help="너무 적으면 정보 손실, 너무 많으면 노이즈. 보통 20~50.")
        with c2:
            sel_method = st.radio(
                "선택 방법",
                ["F-test (빠름)", "Mutual Info (느림·정확)"],
                help="F-test: 선형 관계. Mutual Info: 비선형 관계까지 탐지."
            )

    # ── ★ 불균형 처리 (NEW) ──────────────────────────────────────
    st.markdown("### 4️⃣ 불균형 처리 ⚖️")
    st.caption(
        "소수 클래스(불량) 비율이 10% 미만이면 `class_weight=balanced` 만으로는 부족합니다. "
        "데이터 레벨에서 균형을 맞추거나 결정 임계값을 낮춰야 Recall 이 올라갑니다."
    )
    ib1, ib2 = st.columns([1, 1])
    with ib1:
        resample_method = st.radio(
            "리샘플링 방법",
            ["없음 (class_weight 만)", "SMOTE (소수 합성↑)", "Undersample (다수↓)"],
            index=1,
            help=(
                "• 없음: 데이터 그대로. RF 는 class_weight=balanced, "
                "XGB 는 scale_pos_weight 자동 적용.\n"
                "• SMOTE: 소수 클래스 가까운 점 사이를 보간해서 합성 샘플 생성. "
                "Precision 유지하며 Recall 개선.\n"
                "• Undersample: 다수 클래스 무작위 축소. Recall 대폭 ↑ 하지만 "
                "FP(잘못된 경보) 도 증가. 정보 손실 있음."
            )
        )
    with ib2:
        threshold = st.slider(
            "결정 임계값 (소수 클래스로 판정할 확률 컷오프)",
            0.10, 0.50, 0.50, 0.05,
            help=(
                "기본 0.50. 값을 낮추면 더 많은 케이스를 불량으로 분류 → "
                "Recall ↑, Precision ↓. 제조 현장처럼 FN(놓친 불량) 비용이 큰 경우 "
                "0.2~0.3 으로 낮추는 게 일반적."
            )
        )

    st.markdown("### 5️⃣ 모델 선택")
    ca, cb, cc = st.columns(3)
    with ca: use_rf  = st.checkbox("🌳 Random Forest", True)
    with cb: use_xgb = st.checkbox("⚡ XGBoost", True)
    with cc: use_mlp = st.checkbox("🧠 MLP", False)
    n_est = st.slider("n_estimators", 50, 500, 200, 50)

    if st.button("🚀 학습 시작", type="primary"):
        if not (use_rf or use_xgb or use_mlp):
            st.error("모델을 선택하세요."); st.stop()

        with st.spinner("학습 중..."):
            y_raw = df[target_col]
            X_raw = df[selected]
            X, dropped = clean_numeric_X(X_raw)
            if X.shape[1] == 0:
                st.error("수치형 피처가 없습니다."); st.stop()

            from sklearn.preprocessing import LabelEncoder
            mask = ~y_raw.isna()
            X = X.loc[mask].reset_index(drop=True)
            y_series = y_raw.loc[mask].reset_index(drop=True)
            le = LabelEncoder()
            y = le.fit_transform(y_series.astype(str))
            class_names = list(le.classes_)
            mapping = {orig: int(enc) for enc, orig in enumerate(class_names)}
            st.info("🔖 **클래스 매핑**: " + ", ".join([f"`{k}`→{v}" for k, v in mapping.items()]))
            if len(np.unique(y)) < 2:
                st.error("클래스가 2개 이상 필요."); st.stop()

            # ── ★ 자동 피처 선택 적용 ─────────────────────────────
            if use_auto and top_k and top_k < X.shape[1]:
                from sklearn.feature_selection import (
                    SelectKBest, f_classif, mutual_info_classif
                )
                score_fn = f_classif if "F-test" in sel_method else mutual_info_classif
                selector = SelectKBest(score_fn, k=top_k)
                X_arr = selector.fit_transform(X, y)
                keep_cols = X.columns[selector.get_support()].tolist()
                X = pd.DataFrame(X_arr, columns=keep_cols, index=X.index)
                st.success(
                    f"✅ 자동 선택: {len(keep_cols)}개 피처 사용 "
                    f"(원래 {len(selected)}개 → {len(keep_cols)}개)"
                )

            from sklearn.model_selection import train_test_split
            from collections import Counter
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                st.warning("stratify 생략됨.")

            # ── ★ 리샘플링 적용 (NEW) ─────────────────────────────
            # 주의: 리샘플링은 train set 에만 적용. test set 은 원본 그대로.
            resample_applied = "없음"
            if "SMOTE" in resample_method:
                try:
                    from imblearn.over_sampling import SMOTE
                    n_min = min(Counter(y_tr).values())
                    k_neighbors = max(1, min(5, n_min - 1))
                    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_tr_arr, y_tr = sm.fit_resample(X_tr, y_tr)
                    X_tr = pd.DataFrame(X_tr_arr, columns=X_tr.columns) \
                           if hasattr(X_tr, "columns") else X_tr_arr
                    resample_applied = f"SMOTE (학습셋 → {dict(Counter(y_tr))})"
                except ImportError:
                    st.error("⚠️ imbalanced-learn 미설치. `pip install imbalanced-learn` 후 사용.")
                    st.stop()
            elif "Undersample" in resample_method:
                try:
                    from imblearn.under_sampling import RandomUnderSampler
                    ru = RandomUnderSampler(random_state=42)
                    X_tr_arr, y_tr = ru.fit_resample(X_tr, y_tr)
                    X_tr = pd.DataFrame(X_tr_arr, columns=X_tr.columns) \
                           if hasattr(X_tr, "columns") else X_tr_arr
                    resample_applied = f"Undersample (학습셋 → {dict(Counter(y_tr))})"
                except ImportError:
                    st.error("⚠️ imbalanced-learn 미설치. `pip install imbalanced-learn` 후 사용.")
                    st.stop()

            # 불균형 비율 계산 (XGBoost scale_pos_weight 용)
            counts = Counter(y_tr)
            n_majority = max(counts.values())
            n_minority = min(counts.values())
            spw = n_majority / max(n_minority, 1)

            # ── ★ 리샘플링 적용 시에는 class_weight/scale_pos_weight 끔 ─────
            # 둘 다 켜면 소수 클래스 가중치가 이중 적용되어 오히려 성능 저하.
            already_balanced = "SMOTE" in resample_method or "Undersample" in resample_method

            # 소수 클래스 인덱스 (이진분류일 때 임계값 적용 대상)
            test_counts_for_minor = Counter(y_te) if len(y_te) else Counter(y_tr)
            minor_class_idx = min(test_counts_for_minor, key=test_counts_for_minor.get) \
                              if len(class_names) == 2 else None

            from sklearn.metrics import (accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         confusion_matrix, classification_report)
            results, preds, models = {}, {}, {}
            def ev(y_t, y_p, n):
                avg = "binary" if n == 2 else "weighted"
                return {"Accuracy": accuracy_score(y_t, y_p),
                        "Precision": precision_score(y_t, y_p, average=avg, zero_division=0),
                        "Recall": recall_score(y_t, y_p, average=avg, zero_division=0),
                        "F1": f1_score(y_t, y_p, average=avg, zero_division=0)}

            def apply_threshold(model, X_test):
                """임계값 기반 예측. 이진분류 + threshold ≠ 0.5 일 때만 적용."""
                if len(class_names) == 2 and abs(threshold - 0.5) > 1e-6 \
                   and minor_class_idx is not None:
                    proba = model.predict_proba(X_test)[:, minor_class_idx]
                    pred = np.where(proba >= threshold, minor_class_idx,
                                    1 - minor_class_idx)
                    return pred
                return model.predict(X_test)

            if use_rf:
                from sklearn.ensemble import RandomForestClassifier
                rf_kwargs = dict(n_estimators=n_est, random_state=42, n_jobs=-1)
                if not already_balanced:
                    rf_kwargs["class_weight"] = "balanced"
                rf = RandomForestClassifier(**rf_kwargs)
                rf.fit(X_tr, y_tr); p = apply_threshold(rf, X_te)
                models["RF"] = rf; preds["RF"] = p
                results["Random Forest"] = ev(y_te, p, len(class_names))

            if use_xgb:
                try:
                    from xgboost import XGBClassifier
                    obj = "binary:logistic" if len(class_names) == 2 else "multi:softprob"
                    xgb_kwargs = dict(n_estimators=n_est, learning_rate=0.1,
                                      max_depth=6, objective=obj, random_state=42,
                                      n_jobs=-1, eval_metric="logloss")
                    # 리샘플링 안 했고 + 이진 + 불균형이면 scale_pos_weight 자동 적용
                    if not already_balanced and len(class_names) == 2 and spw > 2:
                        xgb_kwargs["scale_pos_weight"] = spw
                    xgb = XGBClassifier(**xgb_kwargs)
                    xgb.fit(X_tr, y_tr); p = apply_threshold(xgb, X_te)
                    models["XGBoost"] = xgb; preds["XGBoost"] = p
                    results["XGBoost"] = ev(y_te, p, len(class_names))
                except ImportError:
                    st.warning("xgboost 미설치")

            if use_mlp:
                from sklearn.neural_network import MLPClassifier
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
                mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200,
                                    random_state=42, early_stopping=True)
                mlp.fit(X_tr_s, y_tr); p = apply_threshold(mlp, X_te_s)
                models["MLP"] = mlp; preds["MLP"] = p
                results["MLP"] = ev(y_te, p, len(class_names))

        st.success("🎉 완료!")
        # 적용된 불균형 처리 요약
        st.info(
            f"⚙️ **불균형 처리 설정** — 리샘플링: `{resample_applied}` · "
            f"결정 임계값: `{threshold:.2f}` · "
            f"class_weight/scale_pos_weight: `{'OFF (리샘플링과 중복 방지)' if already_balanced else 'ON (자동)'}`"
        )

        # ────────────────────────────────────────────────────────────
        # 5️⃣ 모델 비교 + 해석
        # ────────────────────────────────────────────────────────────
        st.markdown("### 6️⃣ 모델 비교")
        cmp = pd.DataFrame(results).T.map(lambda x: f"{x*100:.1f}%")
        st.dataframe(cmp, use_container_width=True)

        # 자동 해석 텍스트
        best_model = max(results, key=lambda k: results[k]["F1"])
        best_f1 = results[best_model]["F1"]
        best_recall = results[best_model]["Recall"]
        best_prec = results[best_model]["Precision"]
        best_acc = results[best_model]["Accuracy"]

        if best_f1 == 0:
            verdict = (
                f"🔴 **모든 모델의 F1=0 입니다.** 모델이 소수 클래스(불량) 를 "
                f"전혀 잡지 못하고 있어요.\n\n"
                f"**개선 방법**: ① 자동 피처 선택 K 값을 더 줄여보기 (예: 20~30), "
                f"② F-test 대신 Mutual Info 시도, "
                f"③ 데이터가 너무 적은 클래스라면 추가 수집 필요."
            )
        elif best_recall < 0.3:
            verdict = (
                f"🟡 **최고 모델: {best_model}** (F1 {best_f1*100:.1f}%, Recall {best_recall*100:.1f}%).\n\n"
                f"Recall 이 30% 미만 — 불량 케이스의 70% 이상을 놓치고 있습니다. "
                f"제조 현장에선 위험 (불량품 출하 가능성). "
                f"피처 K 를 조정하거나 다른 변수 조합을 시도해 보세요."
            )
        elif best_recall < 0.6:
            verdict = (
                f"🟢 **최고 모델: {best_model}** (F1 {best_f1*100:.1f}%, "
                f"Recall {best_recall*100:.1f}%, Precision {best_prec*100:.1f}%).\n\n"
                f"실무 활용 가능한 수준. Recall 을 더 올리고 싶다면 "
                f"피처 K 늘리기 또는 모델별 임계값 조정."
            )
        else:
            verdict = (
                f"✅ **최고 모델: {best_model}** (F1 {best_f1*100:.1f}%, "
                f"Recall {best_recall*100:.1f}%, Precision {best_prec*100:.1f}%).\n\n"
                f"양호한 성능입니다."
            )
        st.markdown(verdict)
        st.caption(
            "💡 **데이터마다 답이 다릅니다.** 어떤 피처 K 가 최적인지, 어떤 모델이 1등인지는 "
            "데이터의 노이즈 비율·표본 수·불균형 정도에 따라 다릅니다. "
            "여러 K 값을 시도해보는 것이 표준 절차입니다."
        )

        # ────────────────────────────────────────────────────────────
        # 6️⃣ Feature Importance + 해석 (차트 축소)
        # ────────────────────────────────────────────────────────────
        st.markdown("### 7️⃣ Feature Importance — 어떤 변수가 중요한가?")
        fi_keys = [k for k in ["RF","XGBoost"] if k in models]
        if fi_keys:
            setup_korean_font()   # 폰트 보장
            cols = st.columns(len(fi_keys))
            top_features_per_model = {}
            for i, k in enumerate(fi_keys):
                fi = pd.Series(models[k].feature_importances_, index=X.columns)
                fi_top = fi.sort_values(ascending=True).tail(10)
                top_features_per_model[k] = fi_top.index[::-1].tolist()
                fig, ax = plt.subplots(figsize=(4.2, 2.8))   # ← 축소
                ax.barh(fi_top.index.astype(str), fi_top.values,
                        color="#D89478", edgecolor="white", linewidth=0.5)
                ax.set_title(f"{k} Top 10", fontsize=10)
                ax.tick_params(labelsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout(); cols[i].pyplot(fig)

            # 해석: RF/XGBoost 의 Top 변수 일치도
            if len(fi_keys) == 2:
                rf_top5 = set(top_features_per_model["RF"][:5])
                xgb_top5 = set(top_features_per_model["XGBoost"][:5])
                overlap = rf_top5 & xgb_top5
                if len(overlap) >= 3:
                    st.markdown(
                        f"🟢 **두 모델의 Top 5 중 {len(overlap)}개가 일치합니다** "
                        f"(`{', '.join(list(overlap)[:5])}`). "
                        f"→ 결과 신뢰도 ↑. 이 변수들을 공정 개선 우선순위로."
                    )
                else:
                    st.markdown(
                        f"🟡 **두 모델의 Top 5 중 {len(overlap)}개만 일치합니다.** "
                        f"→ 두 모델이 서로 다른 패턴을 보고 있어요. 데이터를 더 살펴보세요 "
                        f"(상관관계 분석, 분포 비교 등)."
                    )
            st.caption(
                "💡 막대가 길수록 그 변수가 불량 예측에 더 큰 영향. "
                "데이터에 따라 어느 변수가 Top 인지는 완전히 달라집니다."
            )

        # ────────────────────────────────────────────────────────────
        # 7️⃣ Confusion Matrix + 해석 (차트 축소)
        # ────────────────────────────────────────────────────────────
        st.markdown("### 8️⃣ Confusion Matrix — 어디서 틀렸나?")
        setup_korean_font()
        cms = st.columns(len(preds))
        cm_summary = {}
        for i, (n, p) in enumerate(preds.items()):
            cm = confusion_matrix(y_te, p)
            cm_summary[n] = cm
            fig, ax = plt.subplots(figsize=(3.2, 2.8))   # ← 축소
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_xlabel("예측", fontsize=8); ax.set_ylabel("실제", fontsize=8)
            for r in range(len(class_names)):
                for c in range(len(class_names)):
                    col = "white" if cm[r,c] > cm.max()/2 else "black"
                    ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                            color=col, fontweight="bold", fontsize=11)
            ax.set_title(n, fontsize=10)
            fig.tight_layout(); cms[i].pyplot(fig)

        # 자동 해석 — 이진분류일 때
        if len(class_names) == 2:
            # 소수 클래스 인덱스 찾기 (불량으로 가정)
            test_counts = Counter(y_te)
            minor_idx = min(test_counts, key=test_counts.get)
            major_idx = 1 - minor_idx
            minor_name = class_names[minor_idx]
            major_name = class_names[major_idx]

            lines = ["**모델별 오분류 분석**:"]
            for n, cm in cm_summary.items():
                fn = cm[minor_idx, major_idx]   # 불량인데 정상으로 예측
                fp = cm[major_idx, minor_idx]   # 정상인데 불량으로 예측
                tp = cm[minor_idx, minor_idx]
                tn = cm[major_idx, major_idx]
                total_minor = fn + tp
                catch_rate = tp / max(total_minor, 1) * 100
                lines.append(
                    f"- **{n}**: `{minor_name}` 중 {tp}/{total_minor}개 ({catch_rate:.0f}%) 잡음. "
                    f"놓친 것 (FN) = **{fn}개**, 잘못된 경보 (FP) = {fp}개"
                )
            st.markdown("\n".join(lines))
            st.caption(
                f"💡 **제조 현장에서는 FN(`{minor_name}` 을 `{major_name}` 으로 본 것) 이 가장 치명적입니다** "
                f"— 불량품이 출하되어 고객 클레임으로 이어지기 때문. "
                f"FP 는 정상품을 추가 검사할 비용 정도. "
                f"비즈니스 비용 구조에 따라 어느 모델이 최선인지 다르게 판단해야 합니다."
            )


# ============================================================================
# ② 시계열 · 수요예측
# ============================================================================
elif menu == "② 시계열 · 수요예측":
    st.markdown("""
    <div class="main-header"><h1>② 시계열 · 수요예측</h1>
    <p>Prophet · LSTM</p></div>
    """, unsafe_allow_html=True)

    # ── 코드 보기 ───────────────────────────────────────────────────
    with st.expander("💻 이 분석의 핵심 코드 보기", expanded=False):
        st.code("""
# ─ Prophet 핵심 흐름 ─
from prophet import Prophet

# Prophet 가산모델: y(t) = trend + seasonality + holidays + noise
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(data)                          # data = [ds, y] 두 컬럼만

future = m.make_future_dataframe(periods=90)   # 90일 미래 프레임
forecast = m.predict(future)         # yhat, yhat_lower, yhat_upper

# ─ LSTM 핵심 흐름 (PyTorch) ─
import torch.nn as nn
class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 학습: 과거 SEQ일을 보고 다음 1일 예측
# 미래 예측: 예측값을 입력에 다시 넣어 재귀적으로 N일 예측 → 오차 누적
""", language="python")

    upload = st.file_uploader("시계열 CSV", type=["csv"], key="ts_up")
    if upload is None:
        require_upload("⬆️ CSV 업로드 (날짜 + 값 컬럼 필요).")

    df = smart_read_csv(upload)
    st.success(f"✅ {df.shape[0]:,}행 × {df.shape[1]:,}컬럼")
    with st.expander("미리보기"):
        st.dataframe(df.head(10), use_container_width=True)

    def guess_date(d):
        for c in d.columns:
            if any(k in str(c).lower() for k in ["date","날짜","일자","time","ds","연월일"]):
                return c
        return d.columns[0]

    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox("날짜 컬럼", df.columns.tolist(),
                                index=df.columns.tolist().index(guess_date(df)))
    with c2:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num:
            st.error("수치형 컬럼 없음"); st.stop()
        value_col = st.selectbox("값 컬럼", num, index=len(num)-1)

    c3, c4 = st.columns(2)
    with c3:
        agg_daily = st.checkbox("같은 날짜 여러 행 → 일별 합계", True)
    with c4:
        horizon = st.slider("예측 기간(일)", 30, 365, 90, 30)

    also_lstm = st.checkbox("🧠 LSTM 도 학습", False)

    if st.button("🔮 예측 시작", type="primary"):
        with st.spinner("Prophet 학습..."):
            data = df[[date_col, value_col]].copy()
            data.columns = ["ds", "y"]
            data["y"] = pd.to_numeric(data["y"], errors="coerce")
            data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
            data = data.replace([np.inf,-np.inf], np.nan).dropna()
            if agg_daily:
                data["ds"] = data["ds"].dt.normalize()
                data = data.groupby("ds", as_index=False)["y"].sum()
            data = data.sort_values("ds").reset_index(drop=True)
            if len(data) < 30:
                st.error(f"30개 이상 필요 (현재 {len(data)})"); st.stop()

            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                        daily_seasonality=False)
            m.fit(data)
            fut = m.make_future_dataframe(periods=horizon)
            fc = m.predict(fut)

            tr_p = fc.iloc[:len(data)]["yhat"].values
            tr_t = data["y"].values
            mae = mean_absolute_error(tr_t, tr_p)
            rmse = np.sqrt(mean_squared_error(tr_t, tr_p))
            den = np.where(tr_t == 0, 1, tr_t)
            mape = np.mean(np.abs((tr_t - tr_p) / den)) * 100

        st.success("🎉 완료!")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}"); c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")

        setup_korean_font()    # ★ 폰트 적용
        fig, ax = plt.subplots(figsize=(11,4.5))
        ax.plot(data["ds"], data["y"], color="#1E3A5F", label="실제")
        ax.plot(fc["ds"], fc["yhat"], color="#D89478", label="예측")
        ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                        color="#D89478", alpha=0.15)
        ax.axvline(data["ds"].max(), color="gray", linestyle="--", alpha=0.6)
        ax.legend(); fig.tight_layout(); st.pyplot(fig)

        st.markdown("### 추세 · 계절성 분해")
        st.pyplot(m.plot_components(fc))

        if also_lstm:
            st.markdown("---")
            st.markdown("### 🧠 LSTM (딥러닝) — Prophet 과 비교")
            with st.spinner("LSTM 학습 중..."):
                try:
                    import torch, torch.nn as nn
                    from sklearn.preprocessing import MinMaxScaler
                    s = data["y"].values.astype(np.float32)
                    sc = MinMaxScaler()
                    ss = sc.fit_transform(s.reshape(-1,1)).flatten()
                    SEQ = min(30, len(ss)//3)
                    if SEQ < 5:
                        st.warning("데이터 부족"); st.stop()
                    Xs, ys = [], []
                    for i in range(len(ss)-SEQ):
                        Xs.append(ss[i:i+SEQ]); ys.append(ss[i+SEQ])
                    Xs = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
                    ys = torch.tensor(np.array(ys), dtype=torch.float32)
                    class Tiny(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.l = nn.LSTM(1,32,batch_first=True); self.f = nn.Linear(32,1)
                        def forward(self,x):
                            o,_ = self.l(x); return self.f(o[:,-1,:]).squeeze(-1)
                    mdl = Tiny()
                    opt = torch.optim.Adam(mdl.parameters(), lr=0.01)
                    fn = nn.MSELoss()
                    mdl.train()
                    for _ in range(50):
                        opt.zero_grad(); p = mdl(Xs); l = fn(p, ys); l.backward(); opt.step()
                    mdl.eval()
                    seq = ss[-SEQ:].tolist(); pr = []
                    with torch.no_grad():
                        for _ in range(horizon):
                            ip = torch.tensor(seq[-SEQ:], dtype=torch.float32).reshape(1,SEQ,1)
                            v = mdl(ip).item(); pr.append(v); seq.append(v)
                    pr_real = sc.inverse_transform(np.array(pr).reshape(-1,1)).flatten()
                    fd = pd.date_range(data["ds"].max()+pd.Timedelta(days=1),
                                       periods=horizon, freq="D")

                    setup_korean_font()    # ★ LSTM 그래프 직전에도 폰트 재적용
                    fig2, ax2 = plt.subplots(figsize=(11,4))
                    ax2.plot(data["ds"], data["y"], color="#1E3A5F",
                             label="실제값", linewidth=1.2)
                    ax2.plot(fd, pr_real, color="#6C7A89",
                             label="LSTM 예측", linewidth=1.5)
                    fpp = fc.tail(horizon)
                    ax2.plot(fpp["ds"], fpp["yhat"], "--", color="#D89478",
                             label="Prophet 예측", linewidth=1.5)
                    ax2.set_xlabel("날짜"); ax2.set_ylabel("값")
                    ax2.set_title("LSTM vs Prophet — 동일 기간 예측 비교")
                    ax2.legend(loc="upper left")
                    ax2.spines["top"].set_visible(False)
                    ax2.spines["right"].set_visible(False)
                    fig2.tight_layout(); st.pyplot(fig2)

                    # ── ★ LSTM vs Prophet 차이 해석 ─────────────
                    lstm_mean = float(np.mean(pr_real))
                    prophet_mean = float(np.mean(fpp["yhat"]))
                    diff_pct = abs(lstm_mean - prophet_mean) / max(abs(prophet_mean), 1) * 100
                    st.info(
                        f"📊 **두 모델 예측 평균**: LSTM={lstm_mean:,.1f}, "
                        f"Prophet={prophet_mean:,.1f} (차이 약 {diff_pct:.0f}%)\n\n"
                        f"💡 **두 모델 예측이 달라도 정상입니다** — 모델 작동 원리가 다르기 때문:\n"
                        f"- **Prophet**: 추세 + 주간/연간 계절성을 명시적으로 분해 → 부드러운 곡선, "
                        f"장기 추세를 잘 따름\n"
                        f"- **LSTM**: 데이터에서 직접 패턴 학습. 본 앱은 50 epoch + 작은 모델(은닉 32)로 "
                        f"학습 → 데이터가 적거나 노이즈가 많으면 평균 근처로 수렴하는 경향\n"
                        f"- **재귀 예측**: LSTM 은 자기 예측값을 다시 입력으로 사용 → 멀어질수록 오차 누적\n\n"
                        f"**언제 LSTM 이 유리한가**: 데이터 1000개+, 강한 비선형 패턴, 다변량 입력 가능 시. "
                        f"본 앱 같은 단변량 + 적은 데이터에선 Prophet 이 더 안정적입니다."
                    )
                except ImportError:
                    st.warning("PyTorch 미설치")


# ============================================================================
# ③ Hugging Face
# ============================================================================
elif menu == "③ Hugging Face 데모":
    st.markdown("""
    <div class="main-header"><h1>③ Hugging Face</h1>
    <p>사전학습 모델 즉시 사용 — 학습 없이 한 줄로 SOTA 모델 활용</p></div>
    """, unsafe_allow_html=True)

    st.markdown(
        "**Hugging Face** 는 100만+ 사전학습 모델을 모아놓은 허브입니다. "
        "`pipeline()` 한 줄로 학습 없이 즉시 사용 가능. 첫 실행 시 모델 자동 다운로드 (1~3분)."
    )

    demo = st.radio("데모", ["📝 텍스트 감성", "🖼️ 이미지 분류"], horizontal=True)

    # ──────────────────────────────────────────────────────────────
    # 텍스트 감성 분석
    # ──────────────────────────────────────────────────────────────
    if demo == "📝 텍스트 감성":
        st.markdown(
            "**모델**: `distilbert-base-uncased-finetuned-sst-2-english`  \n"
            "- **DistilBERT**: 구글 BERT 의 경량화 버전 (속도 60% ↑, 성능 97% 유지)\n"
            "- **SST-2 파인튜닝**: 영화 리뷰 6.7만 건으로 학습 → 영문 긍/부정 판별\n"
            "- **활용 예**: 고객 리뷰·클레임·SNS 텍스트 자동 분류\n"
            "- **한계**: 영문 일반 텍스트로 학습 → 제조 도메인 특화는 Fine-tuning 필요"
        )

        with st.expander("💻 코드 보기", expanded=False):
            st.code("""
from transformers import pipeline

# pipeline 한 줄로 SOTA 모델 로딩 (자동 다운로드 + 토크나이저 설정)
clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# 추론 - 입력 리스트 → 결과 리스트
results = clf(["The product quality is excellent.",
               "Defect rate has been too high."])
# → [{'label': 'POSITIVE', 'score': 0.999},
#    {'label': 'NEGATIVE', 'score': 0.998}]
""", language="python")

        txt = st.text_area(
            "영문 텍스트 (한 줄에 하나)",
            height=140,
            placeholder=(
                "The product quality is excellent.\n"
                "Defect rate has been too high this month.\n"
                "Customer service was very helpful."
            ),
        )
        if st.button("🤖 분석", type="primary"):
            if not txt.strip():
                st.warning("입력하세요."); st.stop()
            with st.spinner("모델 로딩 & 추론 중... (첫 실행 1~2분)"):
                try:
                    from transformers import pipeline
                    clf = pipeline("sentiment-analysis",
                                   model="distilbert-base-uncased-finetuned-sst-2-english")
                    lines = [t.strip() for t in txt.split("\n") if t.strip()]
                    r = clf(lines)
                    st.dataframe(pd.DataFrame({
                        "문장": lines,
                        "판정": [x["label"] for x in r],
                        "확신도": [f"{x['score']*100:.1f}%" for x in r],
                    }), use_container_width=True)
                    st.caption("💡 POSITIVE/NEGATIVE 이진 분류. 중립(neutral) 은 출력하지 않음.")
                except Exception as e:
                    st.error(f"실패: {e}")

    # ──────────────────────────────────────────────────────────────
    # 이미지 분류
    # ──────────────────────────────────────────────────────────────
    else:
        st.markdown(
            "**모델**: `google/vit-base-patch16-224` (Vision Transformer)  \n"
            "- **ViT**: 이미지를 16×16 패치로 잘라 Transformer 로 처리 (2021, Google)\n"
            "- **학습 데이터**: ImageNet-21k (1,400만 장) → ImageNet-1k 파인튜닝\n"
            "- **출력**: 1,000개 카테고리 중 Top-K 분류 (개·고양이·자동차 등 일반 객체)\n"
            "- **한계**: ImageNet 일반 객체 학습 → 제조 결함 검출엔 부적합. "
            "현장에선 **YOLOv8 + 자사 결함 사진** 으로 Fine-tuning 필요."
        )

        with st.expander("💻 코드 보기", expanded=False):
            st.code("""
from transformers import pipeline

# ViT 모델 로딩 - 동일한 pipeline 패턴
clf = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
)

# 이미지 입력 → Top-5 분류
results = clf(img, top_k=5)
# → [{'label': 'tabby cat', 'score': 0.78},
#    {'label': 'tiger cat', 'score': 0.15}, ...]

# ─ 제조 현장 적용 (YOLO) ─
from ultralytics import YOLO
model = YOLO("yolov8n.pt")             # 사전학습 가중치
results = model("product.jpg")          # 한 줄 추론
results[0].show()                       # bounding box 표시
""", language="python")

        up = st.file_uploader("이미지", type=["jpg","jpeg","png"], key="hfimg")
        if up is None:
            require_upload("⬆️ 이미지 업로드")
        from PIL import Image
        img = Image.open(up).convert("RGB")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, use_column_width=True)
        with c2:
            if st.button("🤖 분류", type="primary"):
                with st.spinner("모델 로딩 & 추론 중... (첫 실행 1~3분)"):
                    try:
                        from transformers import pipeline
                        clf = pipeline("image-classification",
                                       model="google/vit-base-patch16-224")
                        r = clf(img, top_k=5)
                        st.dataframe(pd.DataFrame({
                            "순위": range(1, len(r)+1),
                            "분류": [x["label"] for x in r],
                            "확신도": [f"{x['score']*100:.1f}%" for x in r],
                        }), use_container_width=True)
                        st.caption(
                            "💡 ImageNet 1,000개 카테고리 중 Top-5. 제조 결함 검출은 "
                            "YOLOv8 + 자사 라벨링 데이터로 Fine-tuning 필수."
                        )
                    except Exception as e:
                        st.error(f"실패: {e}")

