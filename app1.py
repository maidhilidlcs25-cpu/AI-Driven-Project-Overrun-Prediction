import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="POWERGRID Insight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Train 2 Separate Models ---
def train_models_if_not_exists(df):
    """Trains separate Cost and Timeline models if not already saved."""

    cost_file = "cost_model.joblib"
    time_file = "timeline_model.joblib"

    features = [
        'project_type', 'terrain', 'planned_cost_crores',
        'planned_timeline_months', 'material_cost_index',
        'labour_availability_percent', 'vendor', 'season_of_start'
    ]

    categorical_features = ['project_type', 'terrain', 'vendor', 'season_of_start']
    numeric_features = [
        'planned_cost_crores', 'planned_timeline_months',
        'material_cost_index', 'labour_availability_percent'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X = df[features]

    # ---------------- COST MODEL ----------------
    if not os.path.exists(cost_file):
        st.info("Training Cost Overrun Model...")

        y_cost = df['cost_overrun_percent']

        cost_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))
        ])

        cost_model.fit(X, y_cost)
        joblib.dump(cost_model, cost_file)
        st.success("Cost model saved!")

    # --------------- TIMELINE MODEL --------------
    if not os.path.exists(time_file):
        st.info("Training Timeline Overrun Model...")

        y_time = df['timeline_overrun_percent']

        timeline_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=150, random_state=42))
        ])

        timeline_model.fit(X, y_time)
        joblib.dump(timeline_model, time_file)
        st.success("Timeline model saved!")

    return joblib.load(cost_file), joblib.load(time_file)


# --- Load Data ---
try:
    df = pd.read_csv("powergrid_projects.csv")
except FileNotFoundError:
    st.error("Error: 'powergrid_projects.csv' not found.")
    st.stop()

# Load or train models
cost_model, timeline_model = train_models_if_not_exists(df)

# --- Sidebar Inputs ---
st.sidebar.title("⚡ POWERGRID Insight")
st.sidebar.header("Project Input Parameters")

project_type = st.sidebar.selectbox("Project Type", df['project_type'].unique())
terrain = st.sidebar.selectbox("Terrain", df['terrain'].unique())
vendor = st.sidebar.selectbox("Primary Vendor", df['vendor'].unique())
season = st.sidebar.selectbox("Season of Start", df['season_of_start'].unique())

planned_cost = st.sidebar.slider("Planned Cost (Cr)", 10.0, 500.0, 150.0)
planned_timeline = st.sidebar.slider("Planned Timeline (Months)", 6, 60, 24)
material_index = st.sidebar.slider("Material Cost Index", 0.5, 2.0, 1.1, 0.05)
labour_availability = st.sidebar.slider("Labour Availability (%)", 50, 100, 80) / 100

# Build input dataset
input_data = pd.DataFrame({
    'project_type': [project_type],
    'terrain': [terrain],
    'planned_cost_crores': [planned_cost],
    'planned_timeline_months': [planned_timeline],
    'material_cost_index': [material_index],
    'labour_availability_percent': [labour_availability],
    'vendor': [vendor],
    'season_of_start': [season]
})

# --- Main Title ---
st.title("AI-Driven Project Overrun Prediction")
st.markdown(
    "This dashboard predicts **Cost** and **Timeline** overruns for POWERGRID projects using "
    "two dedicated machine learning models."
)

# --- Prediction Button ---
if st.sidebar.button("Predict Overrun Risk", type="primary"):

    # Predict both
    cost_pred = cost_model.predict(input_data)[0]
    time_pred = timeline_model.predict(input_data)[0]

    # Overall classification
    risk_score = (cost_pred + time_pred) / 2

    if risk_score < 15:
        risk_level = "Low Risk"
        risk_color = "green"
    elif risk_score < 30:
        risk_level = "Medium Risk"
        risk_color = "orange"
    else:
        risk_level = "High Risk"
        risk_color = "red"

    st.subheader("Project Risk Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Overall Project Risk", risk_level)
    col2.metric("Cost Overrun Prediction", f"{cost_pred:.2f}%")
    col3.metric("Timeline Overrun Prediction", f"{time_pred:.2f}%")

    st.markdown(
        f"<p style='color:{risk_color}; font-size:1.2em;'>"
        f"This project is classified as <b>{risk_level}</b>."
        f"</p>",
        unsafe_allow_html=True,
    )

    # =====================================================
    #                   TAB-BASED DASHBOARD
    # =====================================================
    tab1, tab2 = st.tabs(["📉 Cost Overrun Model", "⏳ Timeline Overrun Model"])

    # -------------------- COST MODEL TAB --------------------
    with tab1:
        st.subheader("Top 5 Factors Affecting Cost Overrun")

        reg = cost_model.named_steps['regressor']
        pre = cost_model.named_steps['preprocessor']

        numeric = pre.transformers_[0][2]
        categorical = pre.transformers_[1][2]
        encoded_cats = pre.transformers_[1][1].get_feature_names_out(categorical)

        features_all = list(numeric) + list(encoded_cats)

        imp_df = pd.DataFrame({
            "feature": features_all,
            "importance": reg.feature_importances_
        }).sort_values("importance", ascending=False).head(5)

        fig1 = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            title="Top 5 Cost Overrun Factors", template="plotly_white"
        )
        fig1.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig1, use_container_width=True)

    # -------------------- TIMELINE MODEL TAB --------------------
    with tab2:
        st.subheader("Top 5 Factors Affecting Timeline Overrun")

        reg_t = timeline_model.named_steps['regressor']
        pre_t = timeline_model.named_steps['preprocessor']

        numeric_t = pre_t.transformers_[0][2]
        categorical_t = pre_t.transformers_[1][2]
        encoded_cats_t = pre_t.transformers_[1][1].get_feature_names_out(categorical_t)

        features_all_t = list(numeric_t) + list(encoded_cats_t)

        imp_df_t = pd.DataFrame({
            "feature": features_all_t,
            "importance": reg_t.feature_importances_
        }).sort_values("importance", ascending=False).head(5)

        fig2 = px.bar(
            imp_df_t, x="importance", y="feature", orientation="h",
            title="Top 5 Timeline Overrun Factors", template="plotly_white"
        )
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Adjust parameters and click 'Predict Overrun Risk' to see predictions.")
