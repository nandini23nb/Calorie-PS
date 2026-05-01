"""
app.py
Streamlit UI for the AI-Powered Calorie Prediction System.

Run with:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess_single

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "calorie_model.pkl")
EDA_PATH   = os.path.join(BASE_DIR, "models", "eda_plots.png")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "model_results.png")

st.set_page_config(
    page_title="CalorieAI — Calorie Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1c1f26;
        border: 1px solid #2a2f3d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-val { font-size: 2.4rem; font-weight: 700; color: #e8ff47; }
    .metric-lbl { font-size: 0.8rem; color: #6b7280; letter-spacing: 2px; text-transform: uppercase; }
    .result-box {
        background: linear-gradient(135deg, #1c2310, #111318);
        border: 2px solid #e8ff47;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .cal-number { font-size: 5rem; font-weight: 900; color: #e8ff47; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #4df0ff;
        border-bottom: 1px solid #2a2f3d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stSlider [data-baseweb="slider"] { padding: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


artifacts = load_model()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 CalorieAI")
    st.caption("AI-Powered Fitness Prediction System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home & Predict", "📊 EDA Dashboard", "📈 Model Performance"],
        label_visibility="collapsed",
    )

    st.divider()
    if artifacts:
        best = artifacts["best_model_name"]
        metrics = artifacts["metrics"][best]
        st.success(f"✅ Model Loaded")
        st.markdown(f"**Best Model:** `{best}`")
        st.markdown(f"**R² Score:** `{metrics['R²']:.4f}`")
        st.markdown(f"**MAE:** `{metrics['MAE']:.2f} kcal`")
        st.markdown(f"**RMSE:** `{metrics['RMSE']:.2f} kcal`")
        st.markdown(f"**CV R²:** `{metrics['CV_R²_mean']:.4f} ± {metrics['CV_R²_std']:.4f}`")
    else:
        st.error("⚠️ Model not found!\nRun `python train.py` first.")

    st.divider()
    st.caption("Dataset: 15,000 records | Features: 7 raw + 4 engineered")


# ─── Helper Functions ─────────────────────────────────────────────────────────
def intensity_info(cal, duration):
    rate = cal / duration
    if rate < 3:
        return "🔵 Low Intensity", "#4df0ff"
    elif rate < 6:
        return "🟡 Moderate", "#e8ff47"
    elif rate < 9:
        return "🟠 High Intensity", "#ff6b35"
    else:
        return "🔴 Peak Performance", "#ff4d6d"


def calc_bmi(height_cm, weight_kg):
    return weight_kg / ((height_cm / 100) ** 2)


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def hr_zone(hr, age):
    max_hr = 220 - age
    pct = (hr / max_hr) * 100
    if pct < 50:   return "Zone 1 — Very Light"
    elif pct < 60: return "Zone 2 — Light"
    elif pct < 70: return "Zone 3 — Moderate"
    elif pct < 80: return "Zone 4 — Hard"
    else:          return "Zone 5 — Maximum"


# ─── PAGE: Home & Predict ─────────────────────────────────────────────────────
if page == "🏠 Home & Predict":
    st.markdown("# 🔥 CalorieAI — Calorie Prediction System")
    st.markdown("*Enter your physiological parameters to predict calories burned during exercise.*")
    st.divider()

    # KPI strip
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">15K+</div><div class="metric-lbl">Training Records</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">11</div><div class="metric-lbl">Features Used</div></div>', unsafe_allow_html=True)
    with col3:
        r2_disp = f"{artifacts['metrics'][artifacts['best_model_name']]['R²']:.4f}" if artifacts else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-val">{r2_disp}</div><div class="metric-lbl">R² Score</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-val">3</div><div class="metric-lbl">Models Trained</div></div>', unsafe_allow_html=True)

    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)

        gender = st.selectbox("Gender", ["male", "female"])
        age    = st.slider("Age (years)", 20, 79, 30)

        c1, c2 = st.columns(2)
        with c1:
            height = st.number_input("Height (cm)", 140, 220, 170)
        with c2:
            weight = st.number_input("Weight (kg)", 30, 160, 70)

        duration   = st.slider("Exercise Duration (minutes)", 1, 60, 30)
        heart_rate = st.slider("Heart Rate (bpm)", 60, 200, 100)
        body_temp  = st.slider("Body Temperature (°C)", 37.0, 42.0, 40.0, step=0.1)

        predict_btn = st.button("⚡ Predict Calories", type="primary", use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            if not artifacts:
                st.error("Model not loaded. Run `python train.py` first.")
            else:
                row = {
                    "Gender": gender, "Age": age, "Height": float(height),
                    "Weight": float(weight), "Duration": float(duration),
                    "Heart_Rate": float(heart_rate), "Body_Temp": float(body_temp),
                }
                le    = artifacts["label_encoder"]
                model = artifacts["best_model"]
                X_inp = preprocess_single(row, le)

                with st.spinner("Running AI model..."):
                    prediction = model.predict(X_inp)[0]
                    prediction = max(1.0, min(314.0, prediction))

                label, color = intensity_info(prediction, duration)
                bmi = calc_bmi(height, weight)
                zone = hr_zone(heart_rate, age)

                st.markdown(f"""
                <div class="result-box">
                    <div style="color:#6b7280; font-size:0.85rem; letter-spacing:3px; text-transform:uppercase;">Calories Burned</div>
                    <div class="cal-number">{prediction:.1f}</div>
                    <div style="color:#aaa; font-size:1rem; margin-bottom:16px;">kcal / session</div>
                    <div style="font-size:1.2rem; font-weight:700; color:{color};">{label}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Body Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("BMI", f"{bmi:.1f}", bmi_category(bmi))
                m2.metric("HR Zone", zone.split("—")[0].strip())
                m3.metric("Cal/min", f"{prediction/duration:.1f} kcal")

                # Feature bar chart for this prediction
                st.markdown("#### 🔬 Feature Influence")
                if hasattr(model, "feature_importances_"):
                    fi_series = pd.Series(
                        model.feature_importances_,
                        index=artifacts["feature_names"]
                    ).sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    fig.patch.set_facecolor("#1c1f26")
                    ax.set_facecolor("#1c1f26")
                    colors = ["#e8ff47" if i == 0 else "#2a3a4a" for i in range(len(fi_series))]
                    ax.barh(fi_series.index[::-1], fi_series.values[::-1], color=colors[::-1])
                    ax.tick_params(colors="white", labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#2a2f3d")
                    ax.set_xlabel("Importance", color="white", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Add to session history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {
                    "Gender": gender, "Age": age, "Height": height,
                    "Weight": weight, "Duration": duration,
                    "HR": heart_rate, "Temp": body_temp,
                    "Calories": round(prediction, 1), "Intensity": label
                })
        else:
            st.info("👈 Fill in your parameters and click **Predict Calories**.")

    # History
    if "history" in st.session_state and st.session_state.history:
        st.divider()
        st.markdown("### 📋 Prediction History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ─── PAGE: EDA Dashboard ─────────────────────────────────────────────────────
elif page == "📊 EDA Dashboard":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("Insights from the 15,000-record exercise and calorie dataset.")
    st.divider()

    if os.path.exists(EDA_PATH):
        st.image(EDA_PATH, use_container_width=True)
    else:
        st.warning("EDA plots not found. Run `python train.py` to generate them.")

    # Interactive stats table
    st.markdown("### 📋 Dataset Statistics")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "exercise.csv"))
    cal = pd.read_csv(os.path.join(BASE_DIR, "data", "calories.csv"))
    df = df.merge(cal[["User_ID", "Calories"]], on="User_ID").drop("User_ID", axis=1)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("### 🔍 Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)


# ─── PAGE: Model Performance ──────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance")
    st.divider()

    if not artifacts:
        st.error("No model artifacts found. Run `python train.py` first.")
    else:
        metrics = artifacts["metrics"]
        best    = artifacts["best_model_name"]

        # Metrics comparison table
        st.markdown("### 🏆 Model Comparison")
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name,
                "R² Score": f"{m['R²']:.4f}",
                "MAE (kcal)": f"{m['MAE']:.2f}",
                "RMSE (kcal)": f"{m['RMSE']:.2f}",
                "CV R² Mean": f"{m['CV_R²_mean']:.4f}",
                "CV R² Std": f"{m['CV_R²_std']:.4f}",
                "Best?": "✅" if name == best else ""
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if os.path.exists(RESULTS_PATH):
            st.markdown("### 📊 Evaluation Plots")
            st.image(RESULTS_PATH, use_container_width=True)

        # Feature importance table
        st.markdown("### 🔬 Feature Importance")
        best_model = artifacts["best_model"]
        if hasattr(best_model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": artifacts["feature_names"],
                "Importance": best_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fi["Importance %"] = (fi["Importance"] * 100).round(2).astype(str) + "%"
            st.dataframe(fi[["Feature", "Importance %"]], use_container_width=True, hide_index=True)