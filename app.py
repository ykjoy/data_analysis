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
# ② 시계열 · 수요예측
# ============================================================================
elif menu == "② 시계열 · 수요예측":
    st.markdown("""
    <div class="main-header"><h1>② 시계열 · 수요예측</h1>
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
# ③ Hugging Face
# ============================================================================
elif menu == "③ Hugging Face 데모":
    st.markdown("""
    <div class="main-header"><h1>③ Hugging Face</h1>
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

