import datetime
import os

import joblib
import pandas as pd
import streamlit as st

from scheduler import build_plan, hourly_forecast
from train_models import (
    CLASSIFIER_NAMES,
    REGRESSOR_NAMES,
    train_and_save_models,
)


MODEL_PATH = "focusflow_models.joblib"


@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict) and bundle.get("version") == 2:
            return bundle
    return train_and_save_models(MODEL_PATH)


st.set_page_config(page_title="FocusFlow", page_icon="🧠", layout="wide")
st.title("🧠 FocusFlow")
st.caption("ML-powered daily planner for focus and distraction")

models = load_models()

with st.sidebar:
    st.header("Your day")
    sleep = st.slider("Sleep (hours)", 3.0, 10.0, 6.5, step=0.5)
    phone = st.slider("Phone usage (hours)", 0.0, 10.0, 2.5, step=0.5)
    stress = st.slider("Stress", 1, 5, 3)
    energy = st.slider("Energy", 1, 5, 4)
    motivation = st.slider("Motivation", 1, 5, 4)

st.subheader("Model comparison")
regression_tab, classification_tab = st.tabs(
    ["Regression (Productivity)", "Classification (Distraction)"]
)
with regression_tab:
    regression_results = pd.DataFrame(models["regression_metrics"]).T
    regression_results.index = list(REGRESSOR_NAMES.keys())
    st.caption("Held-out test performance. Lower MAE/RMSE and higher R² are better.")
    st.dataframe(regression_results.style.format("{:.3f}"), use_container_width=True)
with classification_tab:
    classification_results = pd.DataFrame(models["classification_metrics"]).T
    classification_results.index = list(CLASSIFIER_NAMES.keys())
    st.caption("Held-out test performance. Higher scores are better.")
    st.dataframe(classification_results.style.format("{:.3f}"), use_container_width=True)

st.subheader("Generate your plan")
selection_col1, selection_col2 = st.columns(2)
with selection_col1:
    selected_regressor_name = st.selectbox(
        "Productivity model", list(REGRESSOR_NAMES.keys())
    )
with selection_col2:
    selected_classifier_name = st.selectbox(
        "Distraction model", list(CLASSIFIER_NAMES.keys())
    )

if st.button("Generate plan", type="primary"):
    state = {
        "sleep_duration": sleep,
        "phone_usage": phone,
        "stress": stress,
        "energy": energy,
        "motivation": motivation,
    }
    forecast = hourly_forecast(
        state,
        datetime.date.today().weekday(),
        models["regressors"][REGRESSOR_NAMES[selected_regressor_name]],
        models["classifiers"][CLASSIFIER_NAMES[selected_classifier_name]],
    )

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric(
        "Average productivity", f"{forecast['productivity'].mean():.0f}/100"
    )
    metric_col2.metric(
        "Average distraction risk", f"{forecast['distraction'].mean():.0%}"
    )
    st.line_chart(forecast.set_index("hour")[["productivity", "distraction"]])

    st.subheader("Best focus hours")
    plan = build_plan(forecast)
    st.dataframe(
        plan.head(5)[["hour", "productivity", "distraction", "focus_score"]]
        .style.format(
            {
                "productivity": "{:.1f}",
                "distraction": "{:.0%}",
                "focus_score": "{:.1f}",
            }
        ),
        use_container_width=True,
    )