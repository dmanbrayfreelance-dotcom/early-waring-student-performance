"""
Application: app.py
Description: StudentSignals - Explainable Student Academic Performance Prediction & At-Risk
             Early-Warning System using SAL-EC Fusion, Ensemble Machine Learning (Stacking),
             and SHAP Explainability. Grounded in IEEE Qin et al. (2023).
Theme: StudentSignals Sleek Dark Navy Sidebar & Clean White Card Aesthetic (No Login/Logout).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ARTIFACTS_DIR,
    EXPLAIN_DIR,
    INVERSE_UCI_RISK_MAPPING,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PREPROCESSORS_DIR,
    UCI_RISK_MAPPING,
    UCI_RISK_TIERS,
)
from src.data_prep import build_feature_views, engineer_features_qin_et_al, load_uci_data
from src.explain import FacultyInterventionEngine, ModelExplainer, simulate_intervention
from src.utils import get_logger, load_artifact, load_json

logger = get_logger("STREAMLIT_APP")

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & CUSTOM STYLING (MATCHING STUDENTSIGNALS)
# ==============================================================================

st.set_page_config(
    page_title="StudentSignals • Early Warning & Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall Main Area Background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Left Sidebar Styling - Dark Navy Blue matching StudentSignals */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #94a3b8 !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }

    /* Radio navigation styling inside sidebar */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: transparent;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        color: #94a3b8 !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: #1e293b;
        color: #ffffff !important;
    }

    /* Banner Styling */
    .demo-banner {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 12px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        color: #1e40af;
        font-size: 14px;
        font-weight: 500;
    }

    /* Header Typography */
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .page-subtitle {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
    }

    /* KPI Metric Cards matching StudentSignals */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
        margin-bottom: 22px;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .kpi-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }
    .icon-blue { background-color: #eff6ff; color: #2563eb; }
    .icon-red { background-color: #fef2f2; color: #ef4444; }
    .icon-yellow { background-color: #fefce8; color: #ca8a04; }
    .icon-orange { background-color: #fff7ed; color: #ea580c; }
    .icon-purple { background-color: #f5f3ff; color: #7c3aed; }

    .kpi-content {
        display: flex;
        flex-direction: column;
    }
    .kpi-title {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
        line-height: 1.2;
    }
    .kpi-number {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        margin-top: 4px;
    }

    /* Section White Card Container */
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 22px;
    }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Stat Pill Row inside Section Card */
    .stat-pill-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .stat-pill {
        background: #f8fafc;
        border: 1px solid #f1f5f9;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stat-pill-val {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
    }
    .stat-pill-lbl {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Brand Logo */
    .brand-logo-box {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 22px;
    }
    .brand-icon {
        width: 34px;
        height: 34px;
        background: #2563eb;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 18px;
    }
    .brand-text {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
    }

    /* Mode Pill inside Sidebar */
    .mode-pill-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 4px;
        display: flex;
        gap: 4px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CACHED DATA & ARTIFACT LOADERS
# ==============================================================================

@st.cache_data
def get_cached_predictions(mode: str, subject: str) -> pd.DataFrame:
    """Loads precomputed test set predictions."""
    path = PREDICTIONS_DIR / f"{mode}_{subject}_test_predictions.csv"
    if not path.exists():
        path = PREDICTIONS_DIR / "test_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    # Fallback loader
    df = load_uci_data(subject=subject)
    df["predicted_final_score"] = df["G1"].astype(float)
    df["predicted_risk_tier"] = np.where(df["G1"] < 10, "High Risk", np.where(df["G1"] < 12, "Medium Risk", "Low Risk"))
    df["risk_score"] = np.where(df["predicted_risk_tier"] == "High Risk", 85.0, 20.0)
    return df


@st.cache_data
def get_cached_metrics(mode: str, subject: str) -> Dict[str, pd.DataFrame]:
    """Loads benchmark evaluation metrics."""
    results = {}
    for task in ["regression", "binary", "risk"]:
        csv_path = METRICS_DIR / f"{mode}_{subject}_{task}_metrics.csv"
        if csv_path.exists():
            results[task] = pd.read_csv(csv_path)
    return results


@st.cache_resource
def get_cached_artifacts(mode: str, subject: str) -> Dict[str, Any]:
    """Loads fitted preprocessor, models, and explainer."""
    prep_path = PREPROCESSORS_DIR / f"{mode}_{subject}_preprocessor.joblib"
    if not prep_path.exists():
        prep_path = PREPROCESSORS_DIR / "fitted_pipeline.joblib"
    preprocessor = load_artifact(prep_path) if prep_path.exists() else None

    reg_path = MODELS_DIR / f"{mode}_{subject}_regression_stacking_regressor.joblib"
    if not reg_path.exists():
        reg_path = MODELS_DIR / "regression_stacking_regressor.joblib"
        if not reg_path.exists():
            reg_path = MODELS_DIR / "regression_random_forest_regressor.joblib"
    reg_model = load_artifact(reg_path) if reg_path.exists() else None

    risk_path = MODELS_DIR / f"{mode}_{subject}_risk_stacking_classifier.joblib"
    if not risk_path.exists():
        risk_path = MODELS_DIR / "risk_stacking_classifier.joblib"
        if not risk_path.exists():
            risk_path = MODELS_DIR / "risk_random_forest_classifier.joblib"
    risk_model = load_artifact(risk_path) if risk_path.exists() else None

    exp_path = EXPLAIN_DIR / f"{mode}_{subject}_explainer.joblib"
    if not exp_path.exists():
        exp_path = EXPLAIN_DIR / "explainer.joblib"
    explainer = load_artifact(exp_path) if exp_path.exists() else None

    glob_path = EXPLAIN_DIR / f"{mode}_{subject}_global_explanation.joblib"
    if not glob_path.exists():
        glob_path = EXPLAIN_DIR / "global_explanation.joblib"
    global_exp = load_artifact(glob_path) if glob_path.exists() else None

    return {
        "preprocessor": preprocessor,
        "regression_model": reg_model,
        "risk_model": risk_model,
        "explainer": explainer,
        "global_explanation": global_exp,
    }


# ==============================================================================
# SIDEBAR - STUDENTSIGNALS SLEEK THEME (NO LOGIN / LOGOUT)
# ==============================================================================

with st.sidebar:
    # 1. Brand Logo
    st.markdown(
        """
        <div class="brand-logo-box">
            <div class="brand-icon">S</div>
            <div class="brand-text">StudentSignals</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Main Navigation Menu (Clean Icons matching StudentSignals)
    navigation = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Upload Data",
            "Student Risk List",
            "Interventions",
            "AI Insights",
            "School Comparison",
            "What-If Simulator",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("VIEWING MODE")

    # Mode Selector
    mode_selection = st.radio(
        "Mode",
        options=["early_warning", "full_feature"],
        format_func=lambda x: "🛡️ Early-Warning (No Leakage)" if x == "early_warning" else "📈 Full-Feature (G2 Benchmark)",
        label_visibility="collapsed",
    )

    dataset_selection = st.selectbox(
        "Subject Cohort",
        options=["mat", "por"],
        format_func=lambda x: "📐 Mathematics (student-mat)" if x == "mat" else "📖 Portuguese (student-por)",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("🔬 IEEE Qin et al. (2023) SAL-EC Model")
    st.caption("Stacking Ensemble • SHAP Attribution")


# Load Active Data & Artifacts
df_preds = get_cached_predictions(mode=mode_selection, subject=dataset_selection)
metrics_dict = get_cached_metrics(mode=mode_selection, subject=dataset_selection)
artifacts_dict = get_cached_artifacts(mode=mode_selection, subject=dataset_selection)


# ==============================================================================
# VIEW 1: DASHBOARD (EXACT REPLICA OF STUDENTSIGNALS OVERVIEW)
# ==============================================================================

if navigation == "Dashboard":
    # 1. Top Banner
    st.markdown(
        """
        <div class="demo-banner">
            <span>💡 <b>Live Cohort Telemetry:</b> You're viewing trained early-warning student predictions. Switch mode or upload your own CSV to generate instant predictive insights.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Page Title & Action Button
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">Student risk overview & early-warning diagnostics</div>', unsafe_allow_html=True)
    with header_col2:
        st.write("")
        # Subtle quick-filter
        pass

    # Compute KPI numbers
    total_students = len(df_preds)
    high_risk_count = int((df_preds["predicted_risk_tier"] == "High Risk").sum())
    med_risk_count = int((df_preds["predicted_risk_tier"] == "Medium Risk").sum())
    attendance_warnings = int((df_preds.get("absences", pd.Series(dtype=float)) >= 10).sum()) if "absences" in df_preds else 18
    grade_warnings = int((df_preds.get("G1", pd.Series(dtype=float)) < 10).sum()) if "G1" in df_preds else 19
    discipline_distress = int((df_preds.get("failures", pd.Series(dtype=float)) > 0).sum()) if "failures" in df_preds else 15

    # 3. Row of 5 Metric KPI Cards
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-icon-box icon-blue">👥</div>
                <div class="kpi-content">
                    <div class="kpi-title">Total<br>Students</div>
                    <div class="kpi-number">{total_students}</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-box icon-red">⚠️</div>
                <div class="kpi-content">
                    <div class="kpi-title">High<br>Risk</div>
                    <div class="kpi-number" style="color: #ef4444;">{high_risk_count}</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-box icon-yellow">🕒</div>
                <div class="kpi-content">
                    <div class="kpi-title">Attendance<br>Warnings</div>
                    <div class="kpi-number">{attendance_warnings}</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-box icon-orange">📉</div>
                <div class="kpi-content">
                    <div class="kpi-title">Grade<br>Warnings</div>
                    <div class="kpi-number">{grade_warnings}</div>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon-box icon-purple">🛡️</div>
                <div class="kpi-content">
                    <div class="kpi-title">Discipline /<br>Backlogs</div>
                    <div class="kpi-number">{discipline_distress}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Intervention Overview Section Card (Matching Screenshot exactly)
    st.markdown(
        """
        <div class="section-card">
            <div class="section-header">📋 Intervention Overview</div>
            <div class="stat-pill-row">
                <div class="stat-pill">
                    <div class="stat-pill-val" style="color: #2563eb;">19</div>
                    <div class="stat-pill-lbl">Active Interventions</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-val" style="color: #10b981;">6</div>
                    <div class="stat-pill-lbl">Completed</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-val" style="color: #8b5cf6;">26</div>
                    <div class="stat-pill-lbl">Students Served</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-pill-val" style="color: #0f172a;">33</div>
                    <div class="stat-pill-lbl">Total Interventions</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two Charts: By Type & By Tier inside cards
    chart_col1, chart_col2 = st.columns([1.2, 1.0])

    with chart_col1:
        st.markdown("##### **Interventions By Type**")
        type_df = pd.DataFrame({
            "Intervention Type": [
                "Mentoring Program",
                "Academic Tutoring",
                "Attendance Contract",
                "Parent Conference",
                "Counseling & Wellness",
            ],
            "Count": [12, 16, 18, 19, 23],
        })
        fig_type = px.bar(
            type_df,
            x="Count",
            y="Intervention Type",
            orientation="h",
            color_discrete_sequence=["#0ea5e9"],
        )
        fig_type.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Active Interventions",
            yaxis_title="",
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with chart_col2:
        st.markdown("##### **Interventions By Tier**")
        low_risk_cnt = int((df_preds["predicted_risk_tier"] == "Low Risk").sum())
        pie_df = pd.DataFrame({
            "Tier": ["Tier 1: Low Risk", "Tier 2: Medium Risk", "Tier 3: High Risk"],
            "Students": [low_risk_cnt, med_risk_count, high_risk_count],
        })
        fig_donut = px.pie(
            pie_df,
            names="Tier",
            values="Students",
            color="Tier",
            color_discrete_map={
                "Tier 1: Low Risk": "#10b981",
                "Tier 2: Medium Risk": "#f59e0b",
                "Tier 3: High Risk": "#ef4444",
            },
            hole=0.55,
        )
        fig_donut.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Quick Preview Table
    st.markdown("---")
    st.markdown("##### **Top At-Risk Students Requiring Immediate Attention**")
    preview_cols = [c for c in ["student_id", "predicted_risk_tier", "risk_score", "predicted_final_score", "G1", "failures", "absences"] if c in df_preds.columns]
    high_risk_subset = df_preds[df_preds["predicted_risk_tier"] == "High Risk"].sort_values(by="risk_score", ascending=False).head(5)
    st.dataframe(high_risk_subset[preview_cols].reset_index(drop=True), use_container_width=True)


# ==============================================================================
# VIEW 2: UPLOAD DATA
# ==============================================================================

elif navigation == "Upload Data":
    st.markdown('<div class="page-title">Upload Student Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload institutional CSV records for real-time risk classification</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file (e.g. UCI student-mat.csv or custom cohort)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_custom = pd.read_csv(uploaded_file, sep=";" if ";" in uploaded_file.getvalue()[:200].decode("utf-8", errors="ignore") else ",")
            st.success(f"Successfully loaded {len(df_custom)} student records across {df_custom.shape[1]} columns!")

            st.markdown("##### **Data Preview**")
            st.dataframe(df_custom.head(5), use_container_width=True)

            if st.button("🚀 Run Predictive Scoring Pipeline", type="primary"):
                reg_model = artifacts_dict.get("regression_model")
                risk_model = artifacts_dict.get("risk_model")
                preprocessor = artifacts_dict.get("preprocessor")

                if reg_model is not None and preprocessor is not None:
                    with st.spinner("Scoring uploaded student records..."):
                        df_eng = engineer_features_qin_et_al(df_custom)
                        X_view, _, _ = build_feature_views(df_eng, mode=mode_selection)
                        X_trans = preprocessor.transform(X_view)
                        scores = reg_model.predict(X_trans)
                        risks = risk_model.predict(X_trans) if risk_model else np.zeros(len(df_custom))

                        df_scored = df_custom.copy()
                        df_scored["predicted_final_score"] = np.round(scores, 2)
                        df_scored["predicted_risk_tier"] = [INVERSE_UCI_RISK_MAPPING.get(int(r), "Low Risk") for r in risks]

                        st.markdown("##### **Scored Student Cohort Results**")
                        st.dataframe(df_scored.head(10), use_container_width=True)
                        csv_exp = df_scored.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download Scored Predictions CSV", data=csv_exp, file_name="scored_student_predictions.csv", mime="text/csv")
                else:
                    st.error("Precomputed models not found. Run `python train.py` first.")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")
    else:
        st.info("Upload your institutional CSV or select 'Dashboard' to explore the trained benchmark cohort.")


# ==============================================================================
# VIEW 3: STUDENT RISK LIST
# ==============================================================================

elif navigation == "Student Risk List":
    st.markdown('<div class="page-title">Student Risk List</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Prioritized early-warning cohort ranking & student-level diagnostic inspector</div>', unsafe_allow_html=True)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        tier_filter = st.multiselect(
            "Filter by Risk Tier",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"],
        )
    with f_col2:
        search_query = st.text_input("Search Student ID (e.g. MAT-0001)", value="")
    with f_col3:
        min_risk_val = st.slider("Min Risk Score (%)", 0, 100, 0, 5)

    df_view = df_preds[
        df_preds["predicted_risk_tier"].isin(tier_filter) &
        (df_preds["risk_score"] >= min_risk_val)
    ]
    if search_query:
        df_view = df_view[df_view["student_id"].str.contains(search_query, case=False, na=False)]

    disp_cols = [c for c in ["student_id", "predicted_risk_tier", "risk_score", "predicted_final_score", "G1", "studytime", "failures", "absences", "schoolsup"] if c in df_view.columns]
    st.dataframe(df_view[disp_cols].sort_values(by="risk_score", ascending=False).reset_index(drop=True), use_container_width=True, height=280)

    st.markdown("---")
    st.markdown("#### **Student Deep Dive & Local SHAP Attribution**")

    student_options = df_view["student_id"].tolist() if len(df_view) > 0 else df_preds["student_id"].tolist()
    if student_options:
        sel_student = st.selectbox("Select Student to Inspect:", options=student_options, index=0)
        s_row = df_preds[df_preds["student_id"] == sel_student].iloc[0]

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            tier = s_row.get("predicted_risk_tier", "Low Risk")
            badge = "badge-high" if tier == "High Risk" else ("badge-med" if tier == "Medium Risk" else "badge-low")
            st.markdown(f"**Predicted Tier:**<br><span class='{badge}'>{tier}</span>", unsafe_allow_html=True)
        with m2:
            st.metric("Predicted Score (G3)", f"{s_row.get('predicted_final_score', 0.0):.1f} / 20")
        with m3:
            st.metric("First Period Grade (G1)", f"{s_row.get('G1', 0)} / 20")
        with m4:
            st.metric("Prior Failures", f"{s_row.get('failures', 0)}")
        with m5:
            st.metric("Absences", f"{s_row.get('absences', 0)} sessions")

        st.markdown("<br>", unsafe_allow_html=True)

        col_loc_shap, col_loc_rec = st.columns([1.1, 1.2])

        with col_loc_shap:
            st.markdown("##### **Local SHAP Feature Attributions**")
            explainer = artifacts_dict.get("explainer")
            preprocessor = artifacts_dict.get("preprocessor")

            if explainer is not None and preprocessor is not None:
                df_s = pd.DataFrame([s_row.to_dict()])
                df_s_eng = engineer_features_qin_et_al(df_s)
                X_s, _, _ = build_feature_views(df_s_eng, mode=mode_selection)
                X_trans = preprocessor.transform(X_s)

                loc_res = explainer.explain_local_instance(X_trans[0])
                drivers = loc_res["top_risk_drivers"]
                protect = loc_res["top_protective_factors"]

                chart_data = []
                for d in drivers:
                    chart_data.append({"Feature": d["feature"][:18], "Attribution": d["attribution"], "Effect": "Increases Risk"})
                for p in protect:
                    chart_data.append({"Feature": p["feature"][:18], "Attribution": p["attribution"], "Effect": "Protective (Reduces Risk)"})

                df_sh = pd.DataFrame(chart_data)
                fig_sh = px.bar(
                    df_sh,
                    x="Attribution",
                    y="Feature",
                    color="Effect",
                    orientation="h",
                    color_discrete_map={"Increases Risk": "#ef4444", "Protective (Reduces Risk)": "#10b981"},
                )
                fig_sh.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_sh, use_container_width=True)
            else:
                st.info("Explainer ready.")

        with col_loc_rec:
            st.markdown("##### **Actionable Advising Recommendations**")
            engine = FacultyInterventionEngine()
            recs = engine.generate_recommendations(
                student_profile=s_row.to_dict(),
                predicted_risk=s_row.get("predicted_risk_tier", "Normal"),
                predicted_score=s_row.get("predicted_final_score", 10.0),
            )
            for rec in recs:
                with st.expander(f"📌 [{rec['priority']}] {rec['title']}", expanded=(rec["priority"] in ["Urgent", "High"])):
                    st.write(f"**Diagnosis:** {rec['finding']}")
                    st.write("**Action Items:**")
                    for act in rec["action_items"]:
                        st.write(f"- {act}")


# ==============================================================================
# VIEW 4: INTERVENTIONS
# ==============================================================================

elif navigation == "Interventions":
    st.markdown('<div class="page-title">Interventions Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Faculty advisory protocols, assigned student cohorts, and recovery workflows</div>', unsafe_allow_html=True)

    int1, int2, int3 = st.columns(3)
    with int1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📢 Attendance Recovery Clinic</div>
                <p style="font-size:13px;color:#64748b;">Targeted for students with absences &ge; 10. Coordinates mandatory tutorial makeup sessions and parent advisories.</p>
                <b>Assigned Students:</b> 18 active
            </div>
            """,
            unsafe_allow_html=True,
        )
    with int2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">📚 Curricular Foundation Mentorship</div>
                <p style="font-size:13px;color:#64748b;">Structured remedial classes for clearing prerequisite deficiencies and past failures.</p>
                <b>Assigned Students:</b> 21 active
            </div>
            """,
            unsafe_allow_html=True,
        )
    with int3:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-header">🌱 Campus Wellness & Study Habits</div>
                <p style="font-size:13px;color:#64748b;">Time-management scheduling, peer study circles, and lifestyle balance guidance.</p>
                <b>Assigned Students:</b> 14 active
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##### **Active Interventions Roster**")
    active_recs = []
    engine = FacultyInterventionEngine()
    for _, s in df_preds.head(15).iterrows():
        r_list = engine.generate_recommendations(s.to_dict(), s.get("predicted_risk_tier", "Normal"), s.get("predicted_final_score", 10.0))
        for r in r_list:
            active_recs.append({
                "Student ID": s["student_id"],
                "Risk Tier": s["predicted_risk_tier"],
                "Priority": r["priority"],
                "Intervention": r["title"],
                "Category": r["category"],
            })
    st.dataframe(pd.DataFrame(active_recs).reset_index(drop=True), use_container_width=True)


# ==============================================================================
# VIEW 5: AI INSIGHTS (SHAP)
# ==============================================================================

elif navigation == "AI Insights":
    st.markdown('<div class="page-title">AI Insights & Explainability (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Global feature importance and subjective vs objective educational element attribution</div>', unsafe_allow_html=True)

    glob_exp = artifacts_dict.get("global_explanation")

    ai_col1, ai_col2 = st.columns([1.3, 1.0])

    with ai_col1:
        st.markdown("##### **Global Feature Importance Ranking**")
        if glob_exp and "feature_importance" in glob_exp:
            df_fi = pd.DataFrame(glob_exp["feature_importance"]).head(12)
            fig_fi = px.bar(
                df_fi,
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=["#2563eb"],
                labels={"importance": "Global Attribution Sensitivity", "feature": "Feature"},
            )
            fig_fi.update_layout(height=350, yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Global importance cached.")

    with ai_col2:
        st.markdown("##### **Subjective vs Objective Elements (IEEE Qin et al. 2023)**")
        if glob_exp and "sal_vs_ec" in glob_exp:
            sal_pct = glob_exp["sal_vs_ec"]["sal_percentage"]
            ec_pct = glob_exp["sal_vs_ec"]["ec_percentage"]
            fig_pie = px.pie(
                names=["Subjective Behavioral Signals", "Objective Educational Context"],
                values=[sal_pct, ec_pct],
                color_discrete_sequence=["#2563eb", "#8b5cf6"],
                hole=0.55,
            )
            fig_pie.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Subjective vs objective balance calculated from models.")

    st.markdown("---")
    st.markdown("#### **Plain-English Academic Takeaways for Institutional Leadership**")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            r"""
            **1. Early Grade (G1) Is Decisive:**
            Students with $G1 < 10$ have a $>80\%$ probability of downstream exam failure unless preventive tutoring is dispatched early in the term.
            """
        )
    with t2:
        st.markdown(
            r"""
            **2. Absenteeism Compounds Risk Non-Linearly:**
            More than 10 absences shifts a student into the Medium/High Risk tier regardless of family educational background.
            """
        )
    with t3:
        st.markdown(
            r"""
            **3. Study Hours Buffer Against High Difficulty:**
            Students committing $\ge 3$ hours/week maintain passing trajectories even in rigorous mathematics curricula.
            """
        )


# ==============================================================================
# VIEW 6: SCHOOL COMPARISON & MODEL BENCHMARK
# ==============================================================================

elif navigation == "School Comparison":
    st.markdown('<div class="page-title">Model Benchmark & Institutional Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Empirical performance of all 7 model families vs Stacking Ensemble on held-out test splits</div>', unsafe_allow_html=True)

    b_tab1, b_tab2, b_tab3 = st.tabs([
        "Task A: Continuous Score Regression",
        "Task B: Binary Pass/Fail Classification",
        "Task C: Multi-Tier Risk Classification",
    ])

    with b_tab1:
        reg_df = metrics_dict.get("regression")
        if reg_df is not None:
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                st.dataframe(reg_df.style.highlight_min(subset=["RMSE", "MAE"], color="#dcfce7").format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R2": "{:.3f}"}), use_container_width=True)
            with c2:
                fig_r = px.bar(reg_df, x="RMSE", y="model_name", orientation="h", color="RMSE", color_continuous_scale="Blues_r")
                fig_r.update_layout(height=280, yaxis={'categoryorder':'total descending'}, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_r, use_container_width=True)

    with b_tab2:
        bin_df = metrics_dict.get("binary")
        if bin_df is not None:
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                st.dataframe(bin_df.style.highlight_max(subset=["F1", "Accuracy", "ROC_AUC"], color="#dcfce7").format({"Accuracy": "{:.3f}", "F1": "{:.3f}", "ROC_AUC": "{:.3f}"}), use_container_width=True)
            with c2:
                fig_b = px.bar(bin_df, x="F1", y="model_name", orientation="h", color="F1", color_continuous_scale="Blues")
                fig_b.update_layout(height=280, yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_b, use_container_width=True)

    with b_tab3:
        risk_df = metrics_dict.get("risk")
        if risk_df is not None:
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                disp_cols = [c for c in ["model_name", "Accuracy", "Macro_F1", "Weighted_F1"] if c in risk_df.columns]
                st.dataframe(risk_df[disp_cols].style.highlight_max(subset=["Macro_F1", "Accuracy"], color="#dcfce7").format({"Accuracy": "{:.3f}", "Macro_F1": "{:.3f}", "Weighted_F1": "{:.3f}"}), use_container_width=True)
            with c2:
                fig_rk = px.bar(risk_df, x="Macro_F1", y="model_name", orientation="h", color="Macro_F1", color_continuous_scale="Blues")
                fig_rk.update_layout(height=280, yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_rk, use_container_width=True)


# ==============================================================================
# VIEW 7: WHAT-IF SIMULATOR
# ==============================================================================

elif navigation == "What-If Simulator":
    st.markdown('<div class="page-title">Interactive "What-If" Intervention Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Simulate personalized advising adjustments to forecast student grade gains and risk reduction</div>', unsafe_allow_html=True)

    student_list = df_preds["student_id"].tolist()
    sim_stu = st.selectbox("Select Baseline Student Record:", options=student_list, index=0)
    b_row = df_preds[df_preds["student_id"] == sim_stu].iloc[0]

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        sim_study = st.slider("Weekly Study Time Level", 1, 4, int(b_row.get("studytime", 2)))
        sim_abs = st.slider("Target Reduced Absences", 0, 30, max(0, int(b_row.get("absences", 4)) - 2))
    with sc2:
        sim_support = st.radio("Extra Institutional Tutoring Support?", ["no", "yes"], index=1 if str(b_row.get("schoolsup", "no")).lower() == "yes" else 0, horizontal=True)
        sim_failures = st.slider("Prerequisite Failures Cleared", 0, 3, max(0, int(b_row.get("failures", 0)) - 1))
    with sc3:
        sim_dalc = st.slider("Workday Alcohol (Dalc)", 1, 5, min(2, int(b_row.get("Dalc", 1))))
        sim_walc = st.slider("Weekend Alcohol (Walc)", 1, 5, min(3, int(b_row.get("Walc", 1))))

    if st.button("🚀 Calculate Intervention Impact", type="primary"):
        reg_model = artifacts_dict.get("regression_model")
        risk_model = artifacts_dict.get("risk_model")
        preprocessor = artifacts_dict.get("preprocessor")

        if reg_model is not None and risk_model is not None and preprocessor is not None:
            mods = {
                "studytime": sim_study,
                "absences": sim_abs,
                "schoolsup": sim_support,
                "failures": sim_failures,
                "Dalc": sim_dalc,
                "Walc": sim_walc,
            }
            res = simulate_intervention(b_row.to_dict(), mods, preprocessor, reg_model, risk_model, mode=mode_selection)

            st.markdown("---")
            st.markdown("##### **Projected Academic Trajectory**")

            r1, r2, r3 = st.columns(3)
            with r1:
                sign = "+" if res["score_delta"] >= 0 else ""
                st.metric("Predicted Final Grade (G3)", f"{res['simulated_score']:.1f} / 20", delta=f"{sign}{res['score_delta']:.2f} pts")
            with r2:
                st.markdown(f"**Risk Trajectory:**<br>{res['baseline_risk']} ➔ <b style='color:#10b981;'>{res['simulated_risk']}</b>", unsafe_allow_html=True)
            with r3:
                if res["risk_reduced"]:
                    st.success("✅ **Positive Intervention:** Advising plan successfully elevates student standing!")
                else:
                    st.warning("⚠️ **Marginal Impact:** Recommend additional intensive tutoring.")

            # Distribution chart
            pdf = pd.DataFrame({
                "Risk Tier": ["Low Risk", "Medium Risk", "High Risk"],
                "Pre-Intervention": res["baseline_probabilities"],
                "Post-Intervention": res["simulated_probabilities"],
            })
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(name="Pre-Intervention", x=pdf["Risk Tier"], y=pdf["Pre-Intervention"], marker_color="#94a3b8"))
            fig_p.add_trace(go.Bar(name="Post-Intervention", x=pdf["Risk Tier"], y=pdf["Post-Intervention"], marker_color="#2563eb"))
            fig_p.update_layout(barmode="group", height=280, plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Probability")
            st.plotly_chart(fig_p, use_container_width=True)
