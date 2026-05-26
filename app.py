# -*- coding: utf-8 -*-
"""
제조 현장을 위한 데이터 분석 입문 — Streamlit 실습 앱
======================================================
실행:  streamlit run app.py
패키지: requirements.txt 참고
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

# ===========================================================
# 한글 폰트 설정
# ===========================================================
def setup_korean_font():
    """OS에 맞는 한글 폰트 자동 설정"""
    import matplotlib.font_manager as fm
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            break
    plt.rcParams["axes.unicode_minus"] = False

setup_korean_font()

# ===========================================================
# 페이지 설정
# ===========================================================
st.set_page_config(
    page_title="제조 데이터 분석 실습",
    page_icon="🏭",
    layout="wide",
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1E3A5F 0%, #4A90A4 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.6rem; }
    .main-header p  { color: #D8E2EA; margin: 0.3rem 0 0 0; font-size: 0.9rem; }
    .metric-card {
        background: #F8F5F0;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #F47C3C;
    }
    div[data-testid="stMetricValue"] { color: #1E3A5F; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ===========================================================
# 사이드바 — 메뉴 선택
# ===========================================================
st.sidebar.title("🏭 제조 데이터 분석")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "🏠 홈",
        "① 분류·불량분석",
        "② 시계열·수요예측",
        "③ Hugging Face 데모",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("💡 데이터가 없다면 각 메뉴의 **샘플 데이터** 옵션을 사용하세요.")
st.sidebar.caption("📚 교재 PPT의 단계를 따라 진행하세요.")


# ============================================================
# 헬퍼 — CSV 읽기 (CP949/UTF-8 자동 처리)
# ============================================================
def smart_read_csv(file):
    """data.go.kr 파일은 CP949, Kaggle은 UTF-8 인 경우가 많음. 자동 시도."""
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file, encoding="utf-8", errors="ignore")


# ============================================================
# 홈 페이지
# ============================================================
if menu == "🏠 홈":
    st.markdown("""
    <div class="main-header">
        <h1>제조 현장을 위한 데이터 분석 입문</h1>
        <p>Hands-on Workshop · 2 hours · Random Forest · Prophet · Hugging Face</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👋 환영합니다")
    st.write("""
    이 앱은 IT 전공이 아닌 **제조 현장 실무자**를 위한 데이터 분석 실습 도구입니다.
    좌측 사이드바에서 메뉴를 선택해 시작하세요.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### ① 분류·불량분석")
        st.write("Random Forest 로 공정 변수 → 불량 여부 예측. 어떤 센서가 가장 큰 영향을 주는지 분석.")
        st.caption("데이터 예시: UCI SECOM (1,567건 × 590 센서)")
    with col2:
        st.markdown("#### ② 시계열·수요예측")
        st.write("Prophet 으로 과거 데이터에서 미래 값을 예측. 추세 · 계절성도 함께 분해.")
        st.caption("데이터 예시: 월별 수요·발전량·생산량")
    with col3:
        st.markdown("#### ③ Hugging Face 데모")
        st.write("사전학습 모델을 한 줄 코드로 가져와 사용. 이미지 / 텍스트 분석 데모.")
        st.caption("모델: DistilBERT (감성) · ResNet (이미지)")

    st.markdown("---")
    st.markdown("### 📋 학습 흐름")
    st.write("""
    1. **데이터 준비** — Kaggle · data.go.kr 에서 다운로드하거나, 앱의 **샘플 데이터 생성** 기능 사용
    2. **모델 학습** — 메뉴별로 **[학습 시작]** 또는 **[예측 시작]** 버튼 클릭
    3. **결과 해석** — 차트와 지표를 함께 보면서 의미를 파악
    4. **딥러닝 비교** — 체크박스로 ML 결과와 DL 결과를 나란히 비교 (선택)
    """)


# ============================================================
# ① 분류·불량분석
# ============================================================
elif menu == "① 분류·불량분석":
    st.markdown("""
    <div class="main-header">
        <h1>① 분류·불량분석 — Random Forest</h1>
        <p>공정 변수로 불량 여부를 예측 — 어떤 변수가 결정적인가?</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 데이터 로딩 ----------
    st.markdown("### 1️⃣ 데이터 업로드")
    use_sample = st.checkbox("샘플 데이터 사용 (SECOM-유사 형태로 자동 생성)", value=True)

    df = None
    if use_sample:
        # SECOM-like synthetic data: 1500행 × 30 sensors + 1 target
        rng = np.random.RandomState(42)
        n = 1500
        n_feat = 30
        X = rng.randn(n, n_feat)
        # 진짜 영향력 있는 변수 = 5개
        true_w = np.zeros(n_feat); true_w[[2, 7, 13, 19, 25]] = [1.2, -0.9, 0.8, -1.1, 0.7]
        logit = X @ true_w + 0.3 * rng.randn(n)
        y = (logit > 1.5).astype(int)  # 약 6~8% 가 불량(1)
        cols = [f"Sensor_{i:03d}" for i in range(n_feat)]
        df = pd.DataFrame(X, columns=cols)
        df["Pass_Fail"] = y
        st.success(f"✅ 샘플 데이터 생성 완료: {n}행 × {n_feat}센서 (불량 비율 {y.mean()*100:.1f}%)")
    else:
        upload = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
        if upload is not None:
            df = smart_read_csv(upload)
            st.success(f"✅ 업로드 완료: {df.shape[0]}행 × {df.shape[1]}컬럼")

    if df is None:
        st.info("⬆️ 위에서 데이터를 준비하세요.")
        st.stop()

    # 데이터 미리보기
    with st.expander("📋 데이터 미리보기 (상위 10행)"):
        st.dataframe(df.head(10), use_container_width=True)

    # ---------- 타겟 선택 ----------
    st.markdown("### 2️⃣ 타겟(예측 대상) 컬럼 선택")
    target_col = st.selectbox("타겟 컬럼", df.columns.tolist(),
                              index=len(df.columns) - 1)

    # 옵션
    col_a, col_b = st.columns(2)
    with col_a:
        also_mlp = st.checkbox("🧠 딥러닝(MLP)도 함께 학습하여 비교", value=False)
    with col_b:
        n_estimators = st.slider("랜덤 포레스트 — 트리 개수", 50, 500, 200, step=50)

    # ---------- 학습 ----------
    if st.button("🚀 학습 시작", type="primary"):
        with st.spinner("학습 중... (수초 ~ 1분 소요)"):
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, confusion_matrix, classification_report,
            )
            from sklearn.preprocessing import LabelEncoder

            # 결측치 간단 처리 (수치형만 사용)
            df_work = df.copy()
            # 타겟 분리
            y_raw = df_work[target_col]
            X = df_work.drop(columns=[target_col])
            # 수치형만 사용 + 결측치는 중앙값으로
            X = X.select_dtypes(include=[np.number])
            X = X.fillna(X.median())
            # 타겟 인코딩
            if y_raw.dtype == object or y_raw.dtype.name == "category":
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
                class_names = list(le.classes_)
            else:
                y = y_raw.values
                class_names = [str(c) for c in sorted(np.unique(y))]

            if len(np.unique(y)) < 2:
                st.error("⚠️ 타겟에 클래스가 2개 이상 있어야 합니다.")
                st.stop()
            if X.shape[1] == 0:
                st.error("⚠️ 수치형 피처가 없습니다.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # ===== Random Forest =====
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=None,
                random_state=42, n_jobs=-1, class_weight="balanced",
            )
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            rf_metrics = {
                "Accuracy":  accuracy_score(y_test, y_pred_rf),
                "Precision": precision_score(y_test, y_pred_rf, average="binary" if len(class_names) == 2 else "weighted", zero_division=0),
                "Recall":    recall_score(y_test, y_pred_rf, average="binary" if len(class_names) == 2 else "weighted", zero_division=0),
                "F1":        f1_score(y_test, y_pred_rf, average="binary" if len(class_names) == 2 else "weighted", zero_division=0),
            }

            # ===== MLP (선택) =====
            mlp_metrics = None
            y_pred_mlp = None
            if also_mlp:
                from sklearn.neural_network import MLPClassifier
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                mlp = MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=200, random_state=42,
                    early_stopping=True,
                )
                mlp.fit(X_train_s, y_train)
                y_pred_mlp = mlp.predict(X_test_s)
                mlp_metrics = {
                    "Accuracy":  accuracy_score(y_test, y_pred_mlp),
                    "Precision": precision_score(y_test, y_pred_mlp, average="binary" if len(class_names) == 2 else "weighted", zero_division=0),
                    "Recall":    recall_score(y_test, y_pred_mlp, average="binary" if len(class_names) == 2 else "weighted", zero_division=0),
                    "F1":        f1_score(y_test, y_pred_mlp, average="binary" if len(class_names) == 2 else "weighted", zero_division=0),
                }

        # ---------- 결과 출력 ----------
        st.success("🎉 학습 완료!")

        # 핵심 지표 카드
        st.markdown("### 3️⃣ 성능 지표 (Test Set)")
        if mlp_metrics is None:
            cols_m = st.columns(4)
            for i, (k, v) in enumerate(rf_metrics.items()):
                cols_m[i].metric(f"{k}", f"{v*100:.1f}%")
        else:
            metric_df = pd.DataFrame({
                "Random Forest": [f"{v*100:.1f}%" for v in rf_metrics.values()],
                "MLP (딥러닝)":  [f"{v*100:.1f}%" for v in mlp_metrics.values()],
            }, index=list(rf_metrics.keys()))
            st.dataframe(metric_df, use_container_width=True)

        st.markdown("---")

        # Feature Importance + Confusion Matrix
        st.markdown("### 4️⃣ 결과 시각화")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🌳 Feature Importance — Top 10**")
            fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.barh(fi.index, fi.values, color="#F47C3C")
            ax1.set_xlabel("Importance")
            ax1.set_title("어떤 변수가 가장 중요한가?")
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            fig1.tight_layout()
            st.pyplot(fig1)
            st.caption("막대가 길수록 그 변수가 불량 예측에 더 큰 영향을 줍니다 → 공정 개선 우선순위")

        with col2:
            st.markdown("**🎯 Confusion Matrix (RF)**")
            cm = confusion_matrix(y_test, y_pred_rf)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            im = ax2.imshow(cm, cmap="Blues")
            ax2.set_xticks(range(len(class_names)))
            ax2.set_yticks(range(len(class_names)))
            ax2.set_xticklabels(class_names)
            ax2.set_yticklabels(class_names)
            ax2.set_xlabel("예측값")
            ax2.set_ylabel("실제값")
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    txtcolor = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax2.text(j, i, str(cm[i, j]), ha="center", va="center", color=txtcolor, fontsize=14, fontweight="bold")
            ax2.set_title("실제 vs 예측")
            fig2.tight_layout()
            st.pyplot(fig2)
            st.caption("대각선 = 맞춘 것 · 비대각선 = 틀린 것. 비대각선이 적을수록 좋음")

        # 상세 리포트
        with st.expander("📊 상세 분류 리포트 (Precision / Recall / F1 — 클래스별)"):
            report = classification_report(y_test, y_pred_rf, target_names=class_names, zero_division=0)
            st.code(report)

        if mlp_metrics is not None:
            with st.expander("🧠 MLP 모델 — Confusion Matrix 도 함께 보기"):
                cm_mlp = confusion_matrix(y_test, y_pred_mlp)
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.imshow(cm_mlp, cmap="Purples")
                ax3.set_xticks(range(len(class_names)))
                ax3.set_yticks(range(len(class_names)))
                ax3.set_xticklabels(class_names)
                ax3.set_yticklabels(class_names)
                ax3.set_xlabel("예측값")
                ax3.set_ylabel("실제값")
                for i in range(len(class_names)):
                    for j in range(len(class_names)):
                        txtcolor = "white" if cm_mlp[i, j] > cm_mlp.max() / 2 else "black"
                        ax3.text(j, i, str(cm_mlp[i, j]), ha="center", va="center", color=txtcolor, fontsize=14, fontweight="bold")
                ax3.set_title("MLP — 실제 vs 예측")
                fig3.tight_layout()
                st.pyplot(fig3)


# ============================================================
# ② 시계열·수요예측
# ============================================================
elif menu == "② 시계열·수요예측":
    st.markdown("""
    <div class="main-header">
        <h1>② 시계열·수요예측 — Prophet</h1>
        <p>과거 데이터에서 미래를 예측 — 추세 · 계절성 분해 포함</p>
    </div>
    """, unsafe_allow_html=True)

    # 데이터 로딩
    st.markdown("### 1️⃣ 데이터 준비")
    use_sample_ts = st.checkbox("샘플 데이터 사용 (월별 수요량 — 자동 생성)", value=True)

    df_ts = None
    if use_sample_ts:
        rng = np.random.RandomState(7)
        # 5년치 일별 데이터 — 추세 + 주간/연간 계절성 + 잡음
        dates = pd.date_range("2021-01-01", "2025-12-31", freq="D")
        n = len(dates)
        trend = np.linspace(100, 200, n)
        weekly = 8 * np.sin(2 * np.pi * np.arange(n) / 7)
        yearly = 30 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        noise = rng.randn(n) * 6
        values = trend + weekly + yearly + noise
        df_ts = pd.DataFrame({"date": dates, "value": values.round(2)})
        st.success(f"✅ 샘플 데이터 생성 완료: {n}일 (2021-01-01 ~ 2025-12-31)")
    else:
        up = st.file_uploader("CSV 파일 업로드 (date, value 형식 권장)", type=["csv"], key="ts_up")
        if up is not None:
            df_ts = smart_read_csv(up)
            st.success(f"✅ 업로드 완료: {df_ts.shape[0]}행")

    if df_ts is None:
        st.info("⬆️ 위에서 데이터를 준비하세요.")
        st.stop()

    with st.expander("📋 데이터 미리보기 (상위 10행)"):
        st.dataframe(df_ts.head(10), use_container_width=True)

    # 컬럼 선택
    st.markdown("### 2️⃣ 날짜·값 컬럼 지정")
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("날짜 컬럼", df_ts.columns.tolist(),
                                index=0)
    with col2:
        # 수치형 컬럼 우선 추천
        numeric_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
        default_value_col = numeric_cols[0] if numeric_cols else df_ts.columns[-1]
        value_col = st.selectbox("값(예측 대상) 컬럼", df_ts.columns.tolist(),
                                 index=df_ts.columns.tolist().index(default_value_col))

    horizon = st.slider("예측 기간 (일)", 30, 365, 90, step=30)
    also_lstm = st.checkbox("🧠 LSTM(딥러닝) 도 함께 학습 — 학습 시간 1~3분 소요", value=False)

    if st.button("🔮 예측 시작", type="primary"):
        with st.spinner("Prophet 모델 학습 중..."):
            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            # 데이터 정리
            data = df_ts[[date_col, value_col]].copy()
            data.columns = ["ds", "y"]
            data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
            data = data.dropna().sort_values("ds").reset_index(drop=True)

            if len(data) < 30:
                st.error("⚠️ 시계열 학습에는 최소 30개 이상의 데이터가 필요합니다.")
                st.stop()

            # Prophet 학습
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            m.fit(data)
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)

            # 학습 데이터에 대한 평가
            train_pred = forecast.iloc[:len(data)]["yhat"].values
            train_true = data["y"].values
            mae = mean_absolute_error(train_true, train_pred)
            rmse = np.sqrt(mean_squared_error(train_true, train_pred))
            mape = np.mean(np.abs((train_true - train_pred) / np.where(train_true == 0, 1, train_true))) * 100

        st.success("🎉 예측 완료!")

        # 핵심 지표
        st.markdown("### 3️⃣ 정확도 (학습 데이터 기준)")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE  (평균 절대 오차)", f"{mae:.2f}")
        c2.metric("RMSE (제곱근 오차)",   f"{rmse:.2f}")
        c3.metric("MAPE (백분율 오차)",   f"{mape:.2f}%")

        st.caption("💡 MAE/RMSE 는 원래 단위 · MAPE 는 비율(%). 상황에 맞는 지표를 선택해서 보세요.")

        # 예측 결과 차트
        st.markdown("### 4️⃣ 예측 결과")
        fig_main, ax_main = plt.subplots(figsize=(11, 4.5))
        ax_main.plot(data["ds"], data["y"], color="#1E3A5F", label="실제값", linewidth=1.2)
        ax_main.plot(forecast["ds"], forecast["yhat"], color="#F47C3C", label="예측값", linewidth=1.5)
        ax_main.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                              color="#F47C3C", alpha=0.15, label="신뢰구간 (80%)")
        # 예측 시작 지점 표시
        last_train_date = data["ds"].max()
        ax_main.axvline(last_train_date, color="gray", linestyle="--", alpha=0.6)
        ax_main.text(last_train_date, ax_main.get_ylim()[1] * 0.95, "  예측 시작",
                     color="gray", fontsize=9)
        ax_main.set_xlabel("날짜")
        ax_main.set_ylabel("값")
        ax_main.set_title(f"{horizon}일 예측")
        ax_main.legend(loc="upper left")
        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)
        fig_main.tight_layout()
        st.pyplot(fig_main)

        # 분해 차트
        st.markdown("### 5️⃣ 추세 · 계절성 분해")
        st.caption("예측값만 보지 마세요 — '왜' 그렇게 나왔는지 함께 보는 게 핵심")
        fig_comp = m.plot_components(forecast)
        st.pyplot(fig_comp)

        # 예측값 테이블
        with st.expander("📋 예측값 상세 테이블 (마지막 30일)"):
            forecast_show = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30).copy()
            forecast_show.columns = ["날짜", "예측값", "하한", "상한"]
            forecast_show["날짜"] = forecast_show["날짜"].dt.strftime("%Y-%m-%d")
            for c in ["예측값", "하한", "상한"]:
                forecast_show[c] = forecast_show[c].round(2)
            st.dataframe(forecast_show, use_container_width=True)

        # ===== LSTM (선택) =====
        if also_lstm:
            st.markdown("---")
            st.markdown("### 🧠 LSTM (딥러닝) — 추가 비교")
            with st.spinner("LSTM 모델 학습 중... (1~3분 소요)"):
                try:
                    import torch
                    import torch.nn as nn
                    from sklearn.preprocessing import MinMaxScaler

                    # 간단한 1-feature LSTM
                    series = data["y"].values.astype(np.float32)
                    scaler = MinMaxScaler()
                    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
                    SEQ = 30
                    Xs, ys = [], []
                    for i in range(len(series_scaled) - SEQ):
                        Xs.append(series_scaled[i:i + SEQ])
                        ys.append(series_scaled[i + SEQ])
                    Xs = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
                    ys = torch.tensor(np.array(ys), dtype=torch.float32)

                    class TinyLSTM(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.lstm = nn.LSTM(1, 32, batch_first=True)
                            self.fc = nn.Linear(32, 1)
                        def forward(self, x):
                            o, _ = self.lstm(x)
                            return self.fc(o[:, -1, :]).squeeze(-1)

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

                    # 예측 (재귀적)
                    model.eval()
                    last_seq = series_scaled[-SEQ:].tolist()
                    preds = []
                    with torch.no_grad():
                        for _ in range(horizon):
                            inp = torch.tensor(last_seq[-SEQ:], dtype=torch.float32).reshape(1, SEQ, 1)
                            p = model(inp).item()
                            preds.append(p)
                            last_seq.append(p)
                    preds_real = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
                    future_dates = pd.date_range(data["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")

                    fig_lstm, ax_lstm = plt.subplots(figsize=(11, 4))
                    ax_lstm.plot(data["ds"], data["y"], color="#1E3A5F", label="실제값", linewidth=1)
                    ax_lstm.plot(future_dates, preds_real, color="#9B59B6", label="LSTM 예측", linewidth=1.5)
                    # Prophet 예측도 함께
                    fut_pp = forecast.tail(horizon)
                    ax_lstm.plot(fut_pp["ds"], fut_pp["yhat"], color="#F47C3C", label="Prophet 예측", linewidth=1.5, linestyle="--")
                    ax_lstm.set_title("LSTM  vs  Prophet — 동일 기간 예측")
                    ax_lstm.legend()
                    ax_lstm.spines["top"].set_visible(False)
                    ax_lstm.spines["right"].set_visible(False)
                    fig_lstm.tight_layout()
                    st.pyplot(fig_lstm)
                    st.caption("LSTM 은 학습에 시간이 더 걸리지만, 비선형·복잡 패턴을 더 잘 잡을 수 있습니다.")
                except ImportError:
                    st.warning("⚠️ PyTorch 가 설치되지 않았습니다. requirements.txt 의 torch 를 설치하세요.")


# ============================================================
# ③ Hugging Face 데모
# ============================================================
elif menu == "③ Hugging Face 데모":
    st.markdown("""
    <div class="main-header">
        <h1>③ Hugging Face — 사전학습 모델 사용해보기</h1>
        <p>전 세계 100만+ 모델을 한 줄 코드로 — 학습 없이 즉시 사용</p>
    </div>
    """, unsafe_allow_html=True)

    demo_type = st.radio(
        "데모 종류 선택",
        ["📝 텍스트 분류 (감성 분석)", "🖼️ 이미지 분류 (제품 사진)"],
        horizontal=True,
    )

    # ---------- 텍스트 데모 ----------
    if demo_type == "📝 텍스트 분류 (감성 분석)":
        st.markdown("""
        **모델**: `distilbert-base-uncased-finetuned-sst-2-english`
        영문 텍스트의 긍정/부정을 자동 판별. 고객 클레임·리뷰 분석 등에 응용 가능.
        """)
        st.markdown("---")
        default_texts = (
            "The product quality is excellent and shipping was fast.\n"
            "Defect rate has been too high this month.\n"
            "Customer service was very helpful and resolved my issue quickly.\n"
            "The component broke after just one day of use."
        )
        text_input = st.text_area("분석할 텍스트 (한 줄에 하나씩)", value=default_texts, height=140)

        if st.button("🤖 분석 시작", type="primary"):
            with st.spinner("모델 다운로드 & 분석 중... (첫 실행 시 모델 다운로드로 1~2분 소요)"):
                try:
                    from transformers import pipeline
                    clf = pipeline("sentiment-analysis",
                                   model="distilbert-base-uncased-finetuned-sst-2-english")
                    lines = [t.strip() for t in text_input.split("\n") if t.strip()]
                    results = clf(lines)
                    df_res = pd.DataFrame({
                        "문장": lines,
                        "판정": [r["label"] for r in results],
                        "확신도": [f"{r['score']*100:.1f}%" for r in results],
                    })
                    st.success("✅ 분석 완료!")
                    st.dataframe(df_res, use_container_width=True)
                    st.caption("""
                    💡 이 결과는 영문 일반 텍스트로 학습된 모델 — 도메인(제조 클레임 등) 특화가 필요하다면
                    내 데이터로 **Fine-tuning** 하면 정확도가 더 올라갑니다.
                    """)
                except Exception as e:
                    st.error(f"⚠️ 모델 로드 실패: {e}")
                    st.info("💡 requirements.txt 의 transformers·torch 설치를 확인하세요. 첫 실행 시 인터넷 연결이 필요합니다.")

    # ---------- 이미지 데모 ----------
    else:
        st.markdown("""
        **모델**: `google/vit-base-patch16-224` (Vision Transformer)
        업로드한 이미지를 1000개 카테고리로 자동 분류. 제품·부품 식별 워크플로 데모용.
        """)
        st.markdown("---")
        up_img = st.file_uploader("이미지 업로드 (jpg / png)", type=["jpg", "jpeg", "png"])

        if up_img is not None:
            from PIL import Image
            img = Image.open(up_img).convert("RGB")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption="입력 이미지", use_column_width=True)
            with col2:
                if st.button("🤖 분류 시작", type="primary"):
                    with st.spinner("모델 다운로드 & 분류 중... (첫 실행 시 1~3분 소요)"):
                        try:
                            from transformers import pipeline
                            clf = pipeline("image-classification",
                                           model="google/vit-base-patch16-224")
                            results = clf(img, top_k=5)
                            st.success("✅ 분류 완료!")
                            df_res = pd.DataFrame({
                                "순위": range(1, len(results) + 1),
                                "분류 결과": [r["label"] for r in results],
                                "확신도": [f"{r['score']*100:.1f}%" for r in results],
                            })
                            st.dataframe(df_res, use_container_width=True)
                            st.caption("""
                            💡 이 모델은 ImageNet (일반 이미지) 으로 학습됨 — 제조 결함 검출은
                            **YOLOv8** + 내 데이터로 학습이 더 적합합니다 (PPT 슬라이드 32 참고).
                            """)
                        except Exception as e:
                            st.error(f"⚠️ 모델 로드 실패: {e}")
        else:
            st.info("⬆️ 이미지를 업로드하면 분류를 시작합니다.")

        st.markdown("---")
        st.markdown("#### 🎯 실제 현장 적용 시 — YOLO 권장")
        st.code("""# 제품 표면 결함 검출 — YOLOv8 + 내 사진 데이터
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("product.jpg")
results[0].show()  # bounding box 가 그려진 결과 표시""", language="python")
        st.caption("위 코드는 실제 회사 사진 데이터로 Fine-tuning 해서 사용하는 것이 정확도가 가장 높습니다.")
