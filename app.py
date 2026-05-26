# -*- coding: utf-8 -*-
"""
================================================================================
 제조 데이터 분석 실습 앱  (Teaching Edition) — v3
================================================================================
변경 이력 (v3)
  • ⑤ 피처 엔지니어링 탭 신규 추가
      - 날짜 분해 / 계절 / 주기성(sin·cos) / 키워드 분류 / Lag / 이동평균
      - 단위 보정(특정일 이후 ×N) / 원-핫 인코딩
      - 처리 결과 CSV 화면에서 직접 다운로드
  • 홈 화면에 "이 앱 소스코드 다운로드" 버튼 추가 (.py / .txt 둘 다)
v2 → v3 외에는 분류·회귀·시계열·HF 메뉴 동일
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
# 한글 폰트
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
        "② 회귀 · 수치예측",
        "③ 시계열 · 수요예측",
        "④ Hugging Face 데모",
        "⑤ 피처 엔지니어링",
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
        <p>Classification · Regression · Time Series · Hugging Face · Feature Engineering</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("좌측 메뉴에서 실습 항목을 선택하세요.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ① 분류 · 불량분석")
        st.write("Random Forest / XGBoost / MLP 동시 비교")
        st.markdown("#### ② 회귀 · 수치예측")
        st.write("Linear (Beta·p-value 통계표) · Ridge · XGBoost")
        st.markdown("#### ③ 시계열 · 수요예측")
        st.write("Prophet · LSTM")
    with col2:
        st.markdown("#### ④ Hugging Face")
        st.write("텍스트 감성 분석 · 이미지 분류")
        st.markdown("#### ⑤ 피처 엔지니어링 ✨ NEW")
        st.write("CSV 업로드 → 컬럼 추가 → 새 CSV 다운로드")

    st.markdown("---")
    st.markdown("### 📥 이 앱 소스코드 다운로드")
    st.caption("현재 실행 중인 app.py 파일을 그대로 받을 수 있습니다.")

    try:
        # 실행 파일 자기 자신을 읽어서 다운로드 제공
        with open(__file__, "r", encoding="utf-8") as f:
            code_text = f.read()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📄 app.py 다운로드",
                data=code_text,
                file_name="app.py",
                mime="text/x-python",
            )
        with c2:
            st.download_button(
                "📝 app.txt 다운로드 (텍스트)",
                data=code_text,
                file_name="app.txt",
                mime="text/plain",
            )
    except Exception as e:
        st.warning(f"소스 파일 자동 로드 실패: {e}")
        st.caption("이 경우 직접 파일을 텍스트 에디터로 열어 저장하세요.")


# ============================================================================
# ① 분류 · 불량분석
# ============================================================================
elif menu == "① 분류 · 불량분석":
    st.markdown("""
    <div class="main-header"><h1>① 분류 · 불량분석</h1>
    <p>Random Forest · XGBoost · MLP — 3 가지 모델 동시 비교</p></div>
    """, unsafe_allow_html=True)

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

    selected = feature_selector_ui(df, target_col, "cls")
    if not selected:
        st.error("⚠️ 피처를 1개 이상 선택하세요."); st.stop()

    st.markdown("### 3️⃣ 모델 선택")
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

            from sklearn.model_selection import train_test_split
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                st.warning("stratify 생략됨.")

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

            if use_rf:
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                            n_jobs=-1, class_weight="balanced")
                rf.fit(X_tr, y_tr); p = rf.predict(X_te)
                models["RF"] = rf; preds["RF"] = p
                results["Random Forest"] = ev(y_te, p, len(class_names))

            if use_xgb:
                try:
                    from xgboost import XGBClassifier
                    obj = "binary:logistic" if len(class_names) == 2 else "multi:softprob"
                    xgb = XGBClassifier(n_estimators=n_est, learning_rate=0.1,
                                        max_depth=6, objective=obj, random_state=42,
                                        n_jobs=-1, eval_metric="logloss")
                    xgb.fit(X_tr, y_tr); p = xgb.predict(X_te)
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
                mlp.fit(X_tr_s, y_tr); p = mlp.predict(X_te_s)
                models["MLP"] = mlp; preds["MLP"] = p
                results["MLP"] = ev(y_te, p, len(class_names))

        st.success("🎉 완료!")
        st.markdown("### 4️⃣ 모델 비교")
        #cmp = pd.DataFrame(results).T.applymap(lambda x: f"{x*100:.1f}%")
        cmp = pd.DataFrame(results).T.map(lambda x: f"{x*100:.1f}")
        st.dataframe(cmp, use_container_width=True)

        st.markdown("### 5️⃣ Feature Importance")
        fi_keys = [k for k in ["RF","XGBoost"] if k in models]
        if fi_keys:
            cols = st.columns(len(fi_keys))
            for i, k in enumerate(fi_keys):
                fi = pd.Series(models[k].feature_importances_, index=X.columns)
                fi = fi.sort_values(ascending=True).tail(10)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.barh(fi.index.astype(str), fi.values, color="#F47C3C")
                ax.set_title(f"{k} Top 10")
                fig.tight_layout(); cols[i].pyplot(fig)

        st.markdown("### 6️⃣ Confusion Matrix")
        cms = st.columns(len(preds))
        for i, (n, p) in enumerate(preds.items()):
            cm = confusion_matrix(y_te, p)
            fig, ax = plt.subplots(figsize=(5,4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
            for r in range(len(class_names)):
                for c in range(len(class_names)):
                    col = "white" if cm[r,c] > cm.max()/2 else "black"
                    ax.text(c, r, str(cm[r,c]), ha="center", va="center",
                            color=col, fontweight="bold")
            ax.set_title(n); fig.tight_layout(); cms[i].pyplot(fig)


# ============================================================================
# ② 회귀 · 수치예측
# ============================================================================
elif menu == "② 회귀 · 수치예측":
    st.markdown("""
    <div class="main-header"><h1>② 회귀 · 수치예측</h1>
    <p>Linear (Beta · p-value) · Ridge · XGBoost</p></div>
    """, unsafe_allow_html=True)

    upload = st.file_uploader("회귀용 CSV", type=["csv"], key="reg_up")
    if upload is None:
        require_upload("⬆️ CSV 를 업로드하세요.")

    df = smart_read_csv(upload)
    st.success(f"✅ {df.shape[0]:,}행 × {df.shape[1]:,}컬럼")
    with st.expander("📋 미리보기"):
        st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("수치형 컬럼이 없습니다."); st.stop()

    target_col = st.selectbox("예측할 숫자 컬럼", numeric_cols,
                              index=len(numeric_cols)-1)

    selected = feature_selector_ui(df, target_col, "reg")
    if not selected:
        st.error("피처를 1개 이상 선택하세요."); st.stop()

    ca, cb, cc = st.columns(3)
    with ca: use_lin = st.checkbox("📐 Linear (통계표)", True)
    with cb: use_rd  = st.checkbox("🛡️ Ridge", True)
    with cc: use_xr  = st.checkbox("⚡ XGBoost", True)

    if st.button("🚀 학습 시작", type="primary"):
        if not (use_lin or use_rd or use_xr):
            st.error("모델 선택 필요"); st.stop()

        with st.spinner("학습 중..."):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            y_raw = pd.to_numeric(df[target_col], errors="coerce")
            X_raw = df[selected]
            X, _ = clean_numeric_X(X_raw)
            if X.shape[1] == 0:
                st.error(
                    "⚠️ **수치형 피처가 0개입니다.**\n\n"
                    "선택된 피처가 모두 문자형이거나 분산 0입니다.\n"
                    "→ **⑤ 피처 엔지니어링** 메뉴에서 컬럼을 먼저 만들어 보세요."
                ); st.stop()
            m = ~y_raw.isna() & np.isfinite(y_raw)
            X = X.loc[m].reset_index(drop=True)
            y = y_raw.loc[m].values.astype(float)
            if len(y) < 10:
                st.error("유효 행 부족"); st.stop()

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

            results, preds, models = {}, {}, {}
            ols_table, ols_summary = None, None

            if use_lin:
                try:
                    import statsmodels.api as sm
                    X_tr_sm = sm.add_constant(X_tr, has_constant="add")
                    X_te_sm = sm.add_constant(X_te, has_constant="add")
                    X_te_sm = X_te_sm.reindex(columns=X_tr_sm.columns, fill_value=0)
                    ols = sm.OLS(y_tr, X_tr_sm).fit()
                    p = ols.predict(X_te_sm)
                    models["Linear"] = ols; preds["Linear"] = p
                    results["Linear"] = (mean_absolute_error(y_te,p),
                                         np.sqrt(mean_squared_error(y_te,p)),
                                         r2_score(y_te,p))
                    tdf = pd.DataFrame({"변수": ols.params.index,
                                        "Beta": ols.params.values,
                                        "Std Err": ols.bse.values,
                                        "t": ols.tvalues.values,
                                        "p-value": ols.pvalues.values})
                    def sig(p):
                        if pd.isna(p): return ""
                        if p < 0.001: return "★★★"
                        if p < 0.01:  return "★★"
                        if p < 0.05:  return "★"
                        if p < 0.10:  return "."
                        return ""
                    tdf["유의성"] = tdf["p-value"].apply(sig)
                    ols_table = tdf
                    ols_summary = {"R²": ols.rsquared, "Adj.R²": ols.rsquared_adj,
                                   "F p": ols.f_pvalue, "N": int(ols.nobs)}
                except ImportError:
                    st.error("pip install statsmodels"); use_lin = False

            if use_rd:
                from sklearn.linear_model import Ridge
                r = Ridge(alpha=1.0); r.fit(X_tr, y_tr); p = r.predict(X_te)
                models["Ridge"] = r; preds["Ridge"] = p
                results["Ridge"] = (mean_absolute_error(y_te,p),
                                    np.sqrt(mean_squared_error(y_te,p)),
                                    r2_score(y_te,p))
            if use_xr:
                try:
                    from xgboost import XGBRegressor
                    xr = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                                      random_state=42, n_jobs=-1)
                    xr.fit(X_tr, y_tr); p = xr.predict(X_te)
                    models["XGBoost"] = xr; preds["XGBoost"] = p
                    results["XGBoost"] = (mean_absolute_error(y_te,p),
                                          np.sqrt(mean_squared_error(y_te,p)),
                                          r2_score(y_te,p))
                except ImportError:
                    st.warning("xgboost 미설치")

        st.success("🎉 완료!")
        st.markdown("### 모델 비교")
        rows = [{"모델": k, "MAE": f"{v[0]:.3f}", "RMSE": f"{v[1]:.3f}", "R²": f"{v[2]:.3f}"}
                for k, v in results.items()]
        st.dataframe(pd.DataFrame(rows).set_index("모델"), use_container_width=True)

        if ols_table is not None:
            st.markdown("---")
            st.markdown("### 📐 Linear Regression — Beta · p-value")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²",      f"{ols_summary['R²']:.4f}")
            c2.metric("Adj. R²", f"{ols_summary['Adj.R²']:.4f}")
            c3.metric("F p",     f"{ols_summary['F p']:.2e}" if not pd.isna(ols_summary['F p']) else "N/A")
            c4.metric("N",       f"{ols_summary['N']:,}")
            disp = ols_table.copy()
            disp["Beta"]    = disp["Beta"].apply(lambda v: f"{v:,.4f}")
            disp["Std Err"] = disp["Std Err"].apply(lambda v: f"{v:,.4f}")
            disp["t"]       = disp["t"].apply(lambda v: f"{v:,.3f}")
            disp["p-value"] = disp["p-value"].apply(lambda v: f"{v:.4f}" if v>=1e-4 else f"{v:.2e}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption("★ p<0.05, ★★ p<0.01, ★★★ p<0.001 — 유의한 변수일수록 별이 많습니다.")

        non_lin = {k: v for k, v in preds.items() if k != "Linear"}
        if non_lin:
            st.markdown("### 예측 vs 실제 (Ridge / XGBoost)")
            cols = st.columns(len(non_lin))
            for i, (k, p) in enumerate(non_lin.items()):
                fig, ax = plt.subplots(figsize=(5,4))
                ax.scatter(y_te, p, alpha=0.5, color="#1E3A5F", s=15)
                lims = [min(y_te.min(), p.min()), max(y_te.max(), p.max())]
                ax.plot(lims, lims, "--", color="#F47C3C")
                ax.set_xlabel("실제"); ax.set_ylabel("예측")
                ax.set_title(f"{k} (R²={results[k][2]:.3f})")
                fig.tight_layout(); cols[i].pyplot(fig)


# ============================================================================
# ③ 시계열 · 수요예측
# ============================================================================
elif menu == "③ 시계열 · 수요예측":
    st.markdown("""
    <div class="main-header"><h1>③ 시계열 · 수요예측</h1>
    <p>Prophet · LSTM</p></div>
    """, unsafe_allow_html=True)

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

        fig, ax = plt.subplots(figsize=(11,4.5))
        ax.plot(data["ds"], data["y"], color="#1E3A5F", label="실제")
        ax.plot(fc["ds"], fc["yhat"], color="#F47C3C", label="예측")
        ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                        color="#F47C3C", alpha=0.15)
        ax.axvline(data["ds"].max(), color="gray", linestyle="--", alpha=0.6)
        ax.legend(); fig.tight_layout(); st.pyplot(fig)

        st.markdown("### 추세 · 계절성 분해")
        st.pyplot(m.plot_components(fc))

        if also_lstm:
            st.markdown("---")
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
                    fig2, ax2 = plt.subplots(figsize=(11,4))
                    ax2.plot(data["ds"], data["y"], color="#1E3A5F", label="실제")
                    ax2.plot(fd, pr_real, color="#6C7A89", label="LSTM")
                    fpp = fc.tail(horizon)
                    ax2.plot(fpp["ds"], fpp["yhat"], "--", color="#F47C3C", label="Prophet")
                    ax2.legend(); fig2.tight_layout(); st.pyplot(fig2)
                except ImportError:
                    st.warning("PyTorch 미설치")


# ============================================================================
# ④ Hugging Face
# ============================================================================
elif menu == "④ Hugging Face 데모":
    st.markdown("""
    <div class="main-header"><h1>④ Hugging Face</h1>
    <p>사전학습 모델 즉시 사용</p></div>
    """, unsafe_allow_html=True)

    demo = st.radio("데모", ["📝 텍스트 감성", "🖼️ 이미지 분류"], horizontal=True)
    if demo == "📝 텍스트 감성":
        txt = st.text_area("영문 텍스트 (한 줄에 하나)", height=140)
        if st.button("🤖 분석", type="primary"):
            if not txt.strip():
                st.warning("입력하세요."); st.stop()
            with st.spinner("로딩..."):
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
                except Exception as e:
                    st.error(f"실패: {e}")
    else:
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
                with st.spinner("로딩..."):
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
                    except Exception as e:
                        st.error(f"실패: {e}")


# ============================================================================
# ⑤ 피처 엔지니어링 (NEW)
# ============================================================================
elif menu == "⑤ 피처 엔지니어링":
    st.markdown("""
    <div class="main-header"><h1>⑤ 피처 엔지니어링</h1>
    <p>CSV 업로드 → 컬럼 추가 → 새 CSV 다운로드</p></div>
    """, unsafe_allow_html=True)

    st.write(
        "회귀나 분류에 쓸 수 있는 **새로운 입력 변수(피처)** 를 화면에서 직접 만들어 봅니다. "
        "예: 신재생에너지 발전량 데이터에서 *월·요일·계절·발전유형·지역·전일 발전량(lag)* 등을 추가."
    )

    upload = st.file_uploader("원본 CSV 업로드", type=["csv"], key="fe_up")
    if upload is None:
        require_upload("⬆️ 가공할 CSV 를 업로드하세요.")

    df = smart_read_csv(upload)
    st.success(f"✅ 원본: {df.shape[0]:,}행 × {df.shape[1]:,}컬럼")
    with st.expander("📋 원본 미리보기"):
        st.dataframe(df.head(10), use_container_width=True)

    work = df.copy()
    cols_all = work.columns.tolist()

    # ─────────────────────────────────────────────────────────
    # ① 단위 보정 (옵션)
    # ─────────────────────────────────────────────────────────
    st.markdown("### 1️⃣ 단위 보정 (선택)")
    st.caption("예: 신재생 데이터처럼 특정 날짜 이후 값이 1000배로 갑자기 변했을 때 보정.")
    use_unitfix = st.checkbox("단위 보정 사용", value=False, key="fe_unit")
    if use_unitfix:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            unit_date_col = st.selectbox("기준 날짜 컬럼", cols_all, key="fe_udcol")
        with c2:
            unit_target_col = st.selectbox("보정할 값 컬럼", cols_all, key="fe_utcol")
        with c3:
            unit_cutoff = st.text_input("기준일 (YYYY-MM-DD) 이후부터", "2025-01-01")
        with c4:
            unit_div = st.number_input("나눌 값", value=1000.0, step=100.0)
        try:
            tmp_dt = pd.to_datetime(work[unit_date_col], errors="coerce")
            cutoff_ts = pd.Timestamp(unit_cutoff)
            mask = tmp_dt >= cutoff_ts
            n_aff = int(mask.sum())
            st.info(f"📌 적용 대상: {n_aff:,}행")
        except Exception as e:
            st.warning(f"미리보기 실패: {e}")

    # ─────────────────────────────────────────────────────────
    # ② 날짜 컬럼 분해
    # ─────────────────────────────────────────────────────────
    st.markdown("### 2️⃣ 날짜 컬럼 분해")
    use_date = st.checkbox("날짜 → year/month/day/weekday/quarter/계절/sin·cos 추가",
                           value=True, key="fe_date")
    date_col = None
    if use_date:
        date_col = st.selectbox("날짜 컬럼", cols_all, key="fe_datecol")

    # ─────────────────────────────────────────────────────────
    # ③ 카테고리 컬럼에서 키워드 추출
    # ─────────────────────────────────────────────────────────
    st.markdown("### 3️⃣ 텍스트 컬럼 → 키워드 분류")
    st.caption("예: '보령 #3태양광' 에서 → 발전유형=태양광, 지역=보령")
    use_kw = st.checkbox("키워드 분류 사용", value=False, key="fe_kw")
    kw_settings = []
    if use_kw:
        kw_col = st.selectbox("분류할 텍스트 컬럼", cols_all, key="fe_kwcol")
        st.markdown("**분류 규칙** (한 줄에 하나, `새컬럼명=키워드1,키워드2,...` 형식)")
        default_rules = (
            "발전유형=수상태양광,태양광,연료전지,풍력,소수력\n"
            "지역=신보령,신서천,보령,서울,세종,인천,제주,서천,여수,괴산,태안,양양"
        )
        rules_text = st.text_area("규칙", value=default_rules, height=100, key="fe_rules")
        # 파싱
        for line in rules_text.strip().split("\n"):
            if "=" not in line:
                continue
            new_col, kws = line.split("=", 1)
            kw_list = [k.strip() for k in kws.split(",") if k.strip()]
            kw_settings.append((new_col.strip(), kw_list))
        if kw_settings:
            st.caption("→ 생성될 컬럼: " + ", ".join([s[0] for s in kw_settings]))

    # ─────────────────────────────────────────────────────────
    # ④ Lag / 이동평균
    # ─────────────────────────────────────────────────────────
    st.markdown("### 4️⃣ Lag · 이동평균 (시계열 피처)")
    use_lag = st.checkbox("lag_1 / lag_7 / ma_7 / ma_30 추가", value=False, key="fe_lag")
    lag_target = None; lag_group = None; lag_datesort = None
    if use_lag:
        c1, c2, c3 = st.columns(3)
        with c1:
            lag_target = st.selectbox("값 컬럼 (lag 대상)", cols_all, key="fe_lagt")
        with c2:
            grp_options = ["(그룹 없음)"] + cols_all
            lag_group_sel = st.selectbox("그룹 컬럼 (예: 발전설비)", grp_options, key="fe_lagg")
            lag_group = None if lag_group_sel == "(그룹 없음)" else lag_group_sel
        with c3:
            lag_datesort = st.selectbox("정렬용 날짜 컬럼", cols_all, key="fe_lagd")

    # ─────────────────────────────────────────────────────────
    # ⑤ 원-핫 인코딩
    # ─────────────────────────────────────────────────────────
    st.markdown("### 5️⃣ 원-핫 인코딩")
    onehot_cols = st.multiselect(
        "원-핫 변환할 카테고리 컬럼 (위에서 새로 만든 컬럼도 선택 가능)",
        cols_all + [s[0] for s in kw_settings] + (["season"] if use_date else []),
        default=[],
        key="fe_oh"
    )

    st.markdown("---")
    run = st.button("🛠️ 컬럼 추가 실행", type="primary")

    if run:
        with st.spinner("처리 중..."):
            log = []

            # ① 단위 보정
            if use_unitfix:
                try:
                    tmp_dt = pd.to_datetime(work[unit_date_col], errors="coerce")
                    mask = tmp_dt >= pd.Timestamp(unit_cutoff)
                    work[unit_target_col] = pd.to_numeric(work[unit_target_col], errors="coerce")
                    n_aff = int(mask.sum())
                    work.loc[mask, unit_target_col] = work.loc[mask, unit_target_col] / unit_div
                    log.append(f"✅ 단위 보정: {n_aff:,}행 ÷ {unit_div}")
                except Exception as e:
                    log.append(f"⚠️ 단위 보정 실패: {e}")

            # ② 날짜 분해
            if use_date and date_col:
                try:
                    dt = pd.to_datetime(work[date_col], errors="coerce")
                    work["year"]       = dt.dt.year
                    work["month"]      = dt.dt.month
                    work["day"]        = dt.dt.day
                    work["weekday"]    = dt.dt.weekday
                    work["quarter"]    = dt.dt.quarter
                    work["dayofyear"]  = dt.dt.dayofyear
                    work["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
                    def _sn(m):
                        if pd.isna(m): return None
                        m = int(m)
                        if m in [3,4,5]: return "봄"
                        if m in [6,7,8]: return "여름"
                        if m in [9,10,11]: return "가을"
                        return "겨울"
                    work["season"] = work["month"].apply(_sn)
                    work["month_sin"]   = np.sin(2*np.pi*work["month"]/12)
                    work["month_cos"]   = np.cos(2*np.pi*work["month"]/12)
                    work["weekday_sin"] = np.sin(2*np.pi*work["weekday"]/7)
                    work["weekday_cos"] = np.cos(2*np.pi*work["weekday"]/7)
                    work["doy_sin"]     = np.sin(2*np.pi*work["dayofyear"]/365)
                    work["doy_cos"]     = np.cos(2*np.pi*work["dayofyear"]/365)
                    log.append(f"✅ 날짜 분해: '{date_col}' → 13개 컬럼 추가")
                except Exception as e:
                    log.append(f"⚠️ 날짜 분해 실패: {e}")

            # ③ 키워드 분류
            if use_kw and kw_settings:
                for new_col, kws in kw_settings:
                    def _match(s, keywords=kws):
                        s = str(s)
                        for k in keywords:
                            if k in s:
                                return k
                        return "기타"
                    work[new_col] = work[kw_col].apply(_match)
                    log.append(f"✅ 키워드 분류: '{new_col}' (값: {work[new_col].nunique()}종)")

            # ④ Lag
            if use_lag and lag_target:
                try:
                    work[lag_target] = pd.to_numeric(work[lag_target], errors="coerce")
                    sort_keys = [lag_group, lag_datesort] if lag_group else [lag_datesort]
                    sort_keys = [c for c in sort_keys if c]
                    if sort_keys:
                        work = work.sort_values(sort_keys).reset_index(drop=True)
                    if lag_group:
                        g = work.groupby(lag_group, group_keys=False)[lag_target]
                    else:
                        g = work[lag_target]
                    work["lag_1"] = g.shift(1)
                    work["lag_7"] = g.shift(7)
                    if lag_group:
                        work["ma_7"]  = work.groupby(lag_group, group_keys=False)[lag_target]\
                                            .apply(lambda s: s.shift(1).rolling(7,  min_periods=3).mean())
                        work["ma_30"] = work.groupby(lag_group, group_keys=False)[lag_target]\
                                            .apply(lambda s: s.shift(1).rolling(30, min_periods=7).mean())
                    else:
                        work["ma_7"]  = work[lag_target].shift(1).rolling(7,  min_periods=3).mean()
                        work["ma_30"] = work[lag_target].shift(1).rolling(30, min_periods=7).mean()
                    log.append(f"✅ Lag/MA: lag_1, lag_7, ma_7, ma_30")
                except Exception as e:
                    log.append(f"⚠️ Lag 실패: {e}")

            # ⑤ 원-핫
            if onehot_cols:
                # 존재하지 않거나 이미 수치형인 컬럼 필터
                valid = [c for c in onehot_cols if c in work.columns]
                if valid:
                    work = pd.get_dummies(work, columns=valid, dtype=int)
                    log.append(f"✅ 원-핫 인코딩: {valid}")

        st.success("🎉 처리 완료!")
        for line in log:
            st.write(line)

        st.markdown(f"### 결과: **{work.shape[0]:,}행 × {work.shape[1]:,}컬럼**")
        st.dataframe(work.head(20), use_container_width=True)

        with st.expander("📋 추가된/변경된 컬럼 전체 목록"):
            new_cols = [c for c in work.columns if c not in df.columns]
            st.write(f"새로 생성: {len(new_cols)}개")
            st.code(", ".join(new_cols) if new_cols else "(없음)")

        # 다운로드 버튼
        st.markdown("### 📥 새 CSV 다운로드")
        csv_bytes = work.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="💾 처리된 CSV 다운로드",
            data=csv_bytes,
            file_name="featured_data.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(
            "✅ 이 CSV 를 **② 회귀** 또는 **① 분류** 메뉴에 다시 업로드하면 "
            "추가된 피처(month, lag_1, 발전유형_태양광 등)를 그대로 사용할 수 있습니다."
        )
