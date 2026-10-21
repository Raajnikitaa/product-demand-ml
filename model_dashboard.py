"""
DemandWise — Model Dashboard
=============================
Standalone Streamlit page that runs the full ML training pipeline and displays
every parameter, metric, coefficient, and evaluation result in one place.

Run with:
    streamlit run model_dashboard.py
"""

import os
import sys
import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_model import (
    run_training_pipeline,
    load_trained_model,
    validate_training_sources,
    get_default_models_dir,
)
from src.data_loader import discover_excel_files, get_default_data_dir
from src.feature_engineering import ALL_MODEL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, SURGE_THRESHOLD_MULTIPLIER
from src.utils import (
    plot_confusion_matrix_chart,
    plot_roc_curve_chart,
    plot_feature_importance_chart,
    plot_chronological_split_diagram,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DemandWise — Model Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background-color: #F8FAFC !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D3B26 0%, #0F4D32 60%, #0A2E1C 100%) !important;
        box-shadow: 4px 0 20px rgba(0,0,0,0.15) !important;
    }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }

    /* Section headings */
    .section-heading {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }

    /* Info card */
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* Tag chip */
    .chip {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 3px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .chip-blue {
        background: #DBEAFE;
        color: #1E40AF;
        border-color: #93C5FD;
    }
    .chip-amber {
        background: #FEF3C7;
        color: #92400E;
        border-color: #FCD34D;
    }

    /* Param table */
    .param-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    .param-table th {
        background: #F1F5F9;
        color: #475569;
        font-weight: 600;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #E2E8F0;
    }
    .param-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
        vertical-align: top;
    }
    .param-table tr:last-child td { border-bottom: none; }
    .param-table tr:hover td { background: #F8FAFC; }

    /* Green number highlight */
    .val-green { color: #15803D; font-weight: 700; }
    .val-red   { color: #DC2626; font-weight: 700; }
    .val-blue  { color: #2563EB; font-weight: 700; }

    div.stButton > button:first-child {
        background: #15803D !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 22px !important;
    }
    div.stButton > button:first-child:hover {
        background: #166534 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 24px 16px 20px 16px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px;">
        <div style="font-size: 26px; margin-bottom: 6px;">🧠</div>
        <h3 style="margin: 0 0 2px 0; font-size: 20px; font-weight: 800; color:#FFFFFF;">Model Dashboard</h3>
        <span style="font-size: 12px; color:rgba(255,255,255,0.5); letter-spacing:0.08em; text-transform:uppercase;">DemandWise ML</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 6px 4px 4px 4px; font-size:11px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:rgba(255,255,255,0.35);">Sections</div>', unsafe_allow_html=True)

    section = st.radio(
        "Section",
        [
            "📋 Model Parameters",
            "📊 Evaluation Metrics",
            "🔢 Coefficients",
            "🗂️ Data Splits",
            "⚙️ Run Training",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="padding: 14px 4px; border-top: 1px solid rgba(255,255,255,0.08); margin-top:16px;">
        <span style="font-size:11px; color:rgba(255,255,255,0.28);">DemandWise · v1.0</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load existing model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return load_trained_model()

pipeline, meta = load_model()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="font-size: 30px; font-weight: 800; color: #0F172A; margin: 0 0 4px 0;">
        🧠 Model Dashboard
    </h1>
    <p style="color: #64748B; font-size: 15px; margin: 0;">
        Full ML pipeline parameters, training configuration, evaluation metrics, and coefficients.
    </p>
</div>
""", unsafe_allow_html=True)

if meta is None:
    st.warning(
        "⚠️ No trained model found. Go to **⚙️ Run Training** in the sidebar to train the pipeline first.",
        icon="⚠️",
    )


# ===========================================================================
# SECTION 1 — MODEL PARAMETERS
# ===========================================================================
if section == "📋 Model Parameters":
    st.markdown('<div class="section-heading">Model Configuration & Parameters</div>', unsafe_allow_html=True)

    selected_weight = meta.get("selected_class_weight", "balanced") if meta else "—"
    class_weight_label = (
        "None (unweighted)" if selected_weight is None else str(selected_weight)
    )
    # ── Algorithm & Solver ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div style="font-size:13px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px;">Algorithm</div>
            <table class="param-table">
                <tr><td><b>Model Type</b></td><td>Logistic Regression (Binary Classifier)</td></tr>
                <tr><td><b>Solver</b></td><td>L-BFGS (Limited-memory BFGS)</td></tr>
                <tr><td><b>Regularization</b></td><td>L2 (Ridge) — C = 1.0</td></tr>
                <tr><td><b>Selected Class Weights</b></td><td>{class_weight_label}</td></tr>
                <tr><td><b>Max Iterations</b></td><td>1000</td></tr>
                <tr><td><b>Tolerance</b></td><td>1e-4</td></tr>
                <tr><td><b>Multi-class</b></td><td>auto (OvR for binary)</td></tr>
                <tr><td><b>Surge Target</b></td><td>Next 7-day average ≥ 1.30 × prior 7-day average; prior baseline must be positive</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        thresh = meta.get("optimal_decision_threshold", "N/A") if meta else "N/A"
        surge_pct = meta.get("surge_threshold_pct", "+30%") if meta else "+30%"
        val_f1 = round(meta.get("validation_f1", 0) * 100, 2) if meta else "—"
        trained_at = meta.get("training_timestamp", "—") if meta else "—"
        dur = meta.get("train_duration_seconds", "—") if meta else "—"

        st.markdown(f"""
        <div class="info-card">
            <div style="font-size:13px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px;">Training Run</div>
            <table class="param-table">
                <tr><td><b>Trained At</b></td><td class="val-blue">{trained_at}</td></tr>
                <tr><td><b>Training Duration</b></td><td>{dur}s</td></tr>
                <tr><td><b>Surge Threshold</b></td><td class="val-green">{surge_pct} above 7-day avg</td></tr>
                <tr><td><b>Optimal Decision Threshold (τ*)</b></td><td class="val-green">{thresh}</td></tr>
                <tr><td><b>Validation F1 (at τ*)</b></td><td class="val-green">{val_f1}%</td></tr>
                <tr><td><b>Forecast Horizon</b></td><td>7 days ahead</td></tr>
                <tr><td><b>Purge Gap</b></td><td>7 days (per split boundary)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Preprocessing Pipeline ──────────────────────────────────────────────
    st.markdown('<div class="section-heading">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <div style="font-size:13px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:12px;">scikit-learn Pipeline Steps</div>
        <table class="param-table">
            <tr>
                <th>Step</th><th>Transformer</th><th>Applied To</th><th>Parameters</th>
            </tr>
            <tr>
                <td><b>1</b></td>
                <td>StandardScaler</td>
                <td>Numeric features</td>
                <td>with_mean=True, with_std=True</td>
            </tr>
            <tr>
                <td><b>2</b></td>
                <td>OneHotEncoder</td>
                <td>Categorical features</td>
                <td>handle_unknown='ignore', sparse_output=False, drop=None</td>
            </tr>
            <tr>
                <td><b>3</b></td>
                <td>LogisticRegression</td>
                <td>All transformed features</td>
                <td>C=1.0, class_weight={class_weight_label!r}, solver='lbfgs', max_iter=1000</td>
            </tr>
        </table>
        <div style="margin-top:12px; font-size:13px; color:#64748B;">
            ⚡ Preprocessing (scaler + encoder) is fitted <b>exclusively on training data</b>.
            Validation and test sets are transformed using training-fitted transformers only.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Features ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Feature Set</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**Numeric Features**")
        for f in NUMERIC_FEATURES:
            st.markdown(f'<span class="chip">{f}</span>', unsafe_allow_html=True)

    with fc2:
        st.markdown("**Categorical Features**")
        for f in CATEGORICAL_FEATURES:
            st.markdown(f'<span class="chip chip-blue">{f}</span>', unsafe_allow_html=True)

    # ── Dataset provenance ──────────────────────────────────────────────────
    if meta:
        st.markdown('<div class="section-heading">Training Data Provenance</div>', unsafe_allow_html=True)
        files = meta.get("dataset_files", [])
        date_min = meta.get("dataset_date_min", "—")
        date_max = meta.get("dataset_date_max", "—")
        total_rows = meta.get("dataset_total_rows", "—")
        file_metas = meta.get("dataset_file_metas", [])

        st.markdown(f"""
        <div class="info-card">
            <table class="param-table">
                <tr><td><b>Date Range</b></td><td class="val-blue">{date_min} → {date_max}</td></tr>
                <tr><td><b>Total Records</b></td><td class="val-green">{total_rows:,} rows</td></tr>
                <tr><td><b>Dataset Files ({len(files)})</b></td><td>{'<br>'.join(f'• {f}' for f in files)}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        if file_metas:
            with st.expander("📂 Per-File Dataset Details"):
                fm_df = pd.DataFrame(file_metas)
                st.dataframe(fm_df, use_container_width=True)

        st.info(meta.get("training_note", ""))


# ===========================================================================
# SECTION 2 — EVALUATION METRICS
# ===========================================================================
elif section == "📊 Evaluation Metrics":
    st.markdown('<div class="section-heading">Holdout Test Set Performance</div>', unsafe_allow_html=True)

    if not meta or "test_metrics" not in meta:
        st.warning("No evaluation data found. Train the model first.")
    else:
        tm = meta["test_metrics"]
        sp = meta.get("split_info", {})
        nb = meta.get("naive_baseline", {})
        val_metrics = meta.get("validation_metrics")

        for warning in meta.get("evaluation_warnings", []):
            st.warning(warning)

        st.caption(
            meta.get(
                "target_definition",
                "Surge target compares the following 7-day mean with the preceding 7-day mean."
            )
        )

        # Test period info
        st.markdown(f"""
        <div class="info-card" style="background:#F0FDF4; border-color:#86EFAC;">
            Evaluated on held-out test set: <b>{sp.get('test_start', '—')} → {sp.get('test_end', '—')}</b>
            ({sp.get('test_rows', 0):,} records).
            Decision threshold fixed at <b>τ* = {meta.get('optimal_decision_threshold', '—')}</b>
            (selected using validation data only). {meta.get('threshold_selection_objective', '')}
        </div>
        """, unsafe_allow_html=True)

        if val_metrics:
            st.markdown('<div class="section-heading">Validation Results (Model Selection)</div>', unsafe_allow_html=True)
            validation_columns = st.columns(6)
            validation_columns[0].metric("Validation Accuracy", f"{val_metrics['accuracy'] * 100:.2f}%")
            validation_columns[1].metric("Validation Precision", f"{val_metrics['precision'] * 100:.2f}%")
            validation_columns[2].metric("Validation Recall", f"{val_metrics['recall'] * 100:.2f}%")
            validation_columns[3].metric("Validation F1", f"{val_metrics['f1_score']:.4f}")
            validation_columns[4].metric(
                "Validation ROC-AUC",
                f"{val_metrics['roc_auc']:.4f}" if val_metrics.get("roc_auc") is not None else "N/A"
            )
            validation_columns[5].metric(
                "Validation PR-AUC",
                f"{val_metrics['pr_auc']:.4f}" if val_metrics.get("pr_auc") is not None else "N/A"
            )

        split_rows = []
        for split_name, prefix in (("Train", "train"), ("Validation", "val"), ("Test", "test")):
            split_rows.append({
                "Partition": split_name,
                "Date range": f"{sp.get(f'{prefix}_start', '—')} – {sp.get(f'{prefix}_end', '—')}",
                "Records": sp.get(f"{prefix}_rows", "—"),
                "Surges": sp.get(f"{prefix}_surge_count", "—"),
                "No surges": sp.get(f"{prefix}_no_surge_count", "—"),
                "Surge %": sp.get(f"{prefix}_surge_pct", "—"),
            })
        st.markdown('<div class="section-heading">Class Distribution by Time Partition</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

        # KPI metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{tm['accuracy'] * 100:.2f}%")
        m2.metric("Precision", f"{tm['precision'] * 100:.2f}%")
        m3.metric("Recall", f"{tm['recall'] * 100:.2f}%")
        m4.metric("F1-Score", f"{tm['f1_score']:.4f}")
        roc_val = f"{tm['roc_auc']:.4f}" if tm.get("roc_auc") is not None else "N/A"
        m5.metric("ROC-AUC", roc_val)
        brier = tm.get("brier_score")
        m6.metric("Brier Score", f"{brier:.4f}" if brier is not None else "N/A")

        # Class distribution
        cd = tm.get("class_distribution", {})
        if cd:
            st.markdown('<div class="section-heading">Class Distribution — Test Set</div>', unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Total Records", f"{cd.get('total', 0):,}")
            d2.metric("Surge Events", f"{cd.get('surge_count', 0):,}")
            d3.metric("No-Surge Records", f"{cd.get('non_surge_count', 0):,}")
            d4.metric("Surge Rate", f"{cd.get('surge_rate_pct', 0):.2f}%")

        # Confusion matrix + ROC
        st.markdown('<div class="section-heading">Confusion Matrix & ROC Curve</div>', unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_cm = plot_confusion_matrix_chart(tm["confusion_matrix"])
            st.plotly_chart(fig_cm, use_container_width=True)
        with ch2:
            curves = meta.get("curves", {})
            if curves.get("available"):
                fig_roc = plot_roc_curve_chart(curves.get("roc", {}), tm.get("roc_auc"))
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC curve unavailable — single class in test set.")

        # Detailed confusion matrix breakdown
        cm = tm["confusion_matrix"]
        st.markdown('<div class="section-heading">Confusion Matrix Breakdown</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
            <table class="param-table">
                <tr><th>Cell</th><th>Count</th><th>Meaning</th></tr>
                <tr><td><b>True Positive (TP)</b></td><td class="val-green">{cm['tp']:,}</td><td>Surge correctly predicted</td></tr>
                <tr><td><b>True Negative (TN)</b></td><td class="val-green">{cm['tn']:,}</td><td>No-surge correctly predicted</td></tr>
                <tr><td><b>False Positive (FP)</b></td><td class="val-amber">{cm['fp']:,}</td><td>False alarm — surge predicted, none occurred</td></tr>
                <tr><td><b>False Negative (FN)</b></td><td class="val-red">{cm['fn']:,}</td><td>Missed surge — surge occurred, not detected</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Threshold sweep (PR curve)
        curves_data = meta.get("curves", {})
        if curves_data.get("available") and "sweep" in curves_data:
            st.markdown('<div class="section-heading">Threshold Sweep — Precision / Recall / F1</div>', unsafe_allow_html=True)
            sweep_df = pd.DataFrame(curves_data["sweep"])
            fig_sweep = go.Figure()
            fig_sweep.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["precision"], name="Precision", line=dict(color="#15803D", width=2)))
            fig_sweep.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["recall"],   name="Recall",    line=dict(color="#2563EB", width=2)))
            fig_sweep.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["f1"],        name="F1-Score",  line=dict(color="#D97706", width=2.5)))
            fig_sweep.add_vline(x=meta.get("optimal_decision_threshold", 0.5), line_dash="dash", line_color="#DC2626",
                                annotation_text=f"τ* = {meta.get('optimal_decision_threshold')}", annotation_position="top right")
            fig_sweep.update_layout(
                title="Metric Sweep over Decision Thresholds",
                xaxis_title="Decision Threshold",
                yaxis_title="Score",
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", size=13),
                height=380,
            )
            st.plotly_chart(fig_sweep, use_container_width=True)

        # Baseline comparison
        st.markdown('<div class="section-heading">Validation-Only Candidate Comparison</div>', unsafe_allow_html=True)
        validation_comparison = meta.get("validation_model_comparison", [])
        if validation_comparison:
            candidate_rows = []
            for item in validation_comparison:
                metrics = item.get("metrics", {})
                candidate_rows.append({
                    "Candidate": item.get("model_name"),
                    "Class weight": item.get("class_weight"),
                    "Threshold": item.get("threshold"),
                    "Accuracy": metrics.get("accuracy"),
                    "Precision": metrics.get("precision"),
                    "Recall": metrics.get("recall"),
                    "F1": metrics.get("f1_score"),
                    "ROC-AUC": metrics.get("roc_auc"),
                    "PR-AUC": metrics.get("pr_auc"),
                    "Confusion matrix": metrics.get("confusion_matrix", {}).get("matrix"),
                })
            st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True, hide_index=True)
            st.caption(meta.get("selected_model_reason", "Candidate choice was based on validation results."))

        st.markdown('<div class="section-heading">Holdout Reference Baselines</div>', unsafe_allow_html=True)
        majority = meta.get("test_majority_baseline", {})
        bench = [
            {
                "Model": meta.get("algorithm", "DemandWise Logistic Regression"),
                "Threshold": f"τ* = {meta.get('optimal_decision_threshold')}",
                "Accuracy": f"{tm['accuracy'] * 100:.2f}%",
                "Precision": f"{tm['precision'] * 100:.2f}%",
                "Recall": f"{tm['recall'] * 100:.2f}%",
                "F1-Score": f"{tm['f1_score']:.4f}",
                "ROC-AUC": f"{tm['roc_auc']:.4f}" if tm.get("roc_auc") is not None else "N/A",
            },
            {
                "Model": "Training-set majority-class baseline",
                "Threshold": "fixed majority class",
                "Accuracy": f"{majority.get('accuracy', 0) * 100:.2f}%",
                "Precision": f"{majority.get('precision', 0) * 100:.2f}%",
                "Recall": f"{majority.get('recall', 0) * 100:.2f}%",
                "F1-Score": f"{majority.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{majority.get('roc_auc'):.4f}" if majority.get("roc_auc") is not None else "N/A",
            },
            {
                "Model": "Naive Momentum Rule",
                "Threshold": f"th = {nb.get('threshold_used', 0.30)}",
                "Accuracy": f"{nb.get('accuracy', 0) * 100:.2f}%",
                "Precision": f"{nb.get('precision', 0) * 100:.2f}%",
                "Recall": f"{nb.get('recall', 0) * 100:.2f}%",
                "F1-Score": f"{nb.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{nb.get('roc_auc', 0):.4f}" if nb.get("roc_auc") else "N/A",
            },
        ]
        st.dataframe(pd.DataFrame(bench), use_container_width=True, hide_index=True)

        # Probability calibration note
        st.markdown('<div class="section-heading">Probability Calibration Note</div>', unsafe_allow_html=True)
        st.info(meta.get("probability_calibration_note", ""))


# ===========================================================================
# SECTION 3 — COEFFICIENTS
# ===========================================================================
elif section == "🔢 Coefficients":
    st.markdown('<div class="section-heading">Logistic Regression Coefficients & Odds Ratios</div>', unsafe_allow_html=True)

    if not meta or "coefficients" not in meta:
        st.warning("No coefficient data found. Train the model first.")
    else:
        coefs = meta["coefficients"]

        # Chart — top 14
        fig_coef = plot_feature_importance_chart(coefs, top_n=14)
        st.plotly_chart(fig_coef, use_container_width=True)

        # Full table
        st.markdown('<div class="section-heading">Full Coefficients Table</div>', unsafe_allow_html=True)
        coef_df = pd.DataFrame(coefs)
        coef_df["abs_coef"] = coef_df["coefficient"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
        coef_df.index = range(1, len(coef_df) + 1)
        st.dataframe(coef_df, use_container_width=True)

        # Interpretation guide
        st.markdown('<div class="section-heading">Interpretation Guide</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <table class="param-table">
                <tr><th>Metric</th><th>Formula</th><th>Interpretation</th></tr>
                <tr>
                    <td><b>Coefficient (β)</b></td>
                    <td>Raw LR coefficient</td>
                    <td>Log-odds change per 1-unit increase in the (scaled) feature. Positive → increases surge likelihood.</td>
                </tr>
                <tr>
                    <td><b>Odds Ratio (e^β)</b></td>
                    <td>exp(β)</td>
                    <td>Multiplicative factor on surge odds. e^β > 1 amplifies odds; e^β < 1 reduces odds.</td>
                </tr>
                <tr>
                    <td><b>Direction</b></td>
                    <td>sign(β)</td>
                    <td>"Increases Surge Odds" if β > 0, "Decreases Surge Odds" if β < 0.</td>
                </tr>
            </table>
            <div style="margin-top:10px; font-size:13px; color:#64748B;">
                ⚠️ Coefficients are on the <b>standardized scale</b>. Numeric features were z-scored before fitting,
                so magnitudes are comparable across numeric features but not directly between numeric and one-hot-encoded categoricals.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# SECTION 4 — DATA SPLITS
# ===========================================================================
elif section == "🗂️ Data Splits":
    st.markdown('<div class="section-heading">Chronological Split Strategy</div>', unsafe_allow_html=True)

    if not meta or "split_info" not in meta:
        st.warning("No split info found. Train the model first.")
    else:
        sp = meta["split_info"]

        # Timeline diagram
        fig_splits = plot_chronological_split_diagram(sp)
        st.plotly_chart(fig_splits, use_container_width=True)

        # Split breakdown table
        st.markdown('<div class="section-heading">Split Statistics</div>', unsafe_allow_html=True)
        split_table = [
            {"Partition": "Training Core",      "Start": sp.get("train_start"), "End": sp.get("train_end"), "Rows": f"{sp.get('train_rows', 0):,}", "Surge %": f"{sp.get('train_surge_pct', 0):.2f}%", "Purpose": "Fit pipeline (scaler + encoder + classifier)"},
            {"Partition": "Purge Gap 1",         "Start": sp.get("val_purge_start"), "End": sp.get("val_purge_end"), "Rows": "—", "Surge %": "—", "Purpose": "7-day buffer preventing forward-label leakage into validation"},
            {"Partition": "Validation Slice",    "Start": sp.get("val_start"), "End": sp.get("val_end"), "Rows": f"{sp.get('val_rows', 0):,}", "Surge %": f"{sp.get('val_surge_pct', 0):.2f}%", "Purpose": "Tune decision threshold τ*"},
            {"Partition": "Purge Gap 2",         "Start": sp.get("test_purge_start"), "End": sp.get("test_purge_end"), "Rows": "—", "Surge %": "—", "Purpose": "7-day buffer preventing forward-label leakage into test"},
            {"Partition": "Holdout Test Set",    "Start": sp.get("test_start"), "End": sp.get("test_end"), "Rows": f"{sp.get('test_rows', 0):,}", "Surge %": f"{sp.get('test_surge_pct', 0):.2f}%", "Purpose": "Final unbiased evaluation (never touched during training)"},
        ]
        st.dataframe(pd.DataFrame(split_table), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-heading">Why Purge Gaps Are Mandatory</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <p style="font-size:14px; color:#334155; line-height:1.7;">
                Each training row at date <b>t</b> has a forward-looking surge target computed from sales in
                <b>[t+1, t+7]</b>. Without a 7-day gap between the training boundary and the validation period,
                the last 7 training rows' targets would use observations from the validation period — this is
                <b>temporal target leakage</b>. The purge gap physically removes those rows from being used near
                the split boundary, preventing any information from "future" periods contaminating the model.
            </p>
            <table class="param-table">
                <tr><th>Guard</th><th>Detail</th></tr>
                <tr><td>Feature Horizon</td><td>All rolling features at date t use only data ≤ t</td></tr>
                <tr><td>Target Horizon</td><td>Surge label uses sales from [t+1, t+7]</td></tr>
                <tr><td>Purge Gap</td><td>7-day buffer between each split boundary</td></tr>
                <tr><td>Preprocessing</td><td>StandardScaler & OneHotEncoder fitted on training only</td></tr>
                <tr><td>Threshold Tuning</td><td>τ* selected on validation only — test never seen</td></tr>
                <tr><td>Final Evaluation</td><td>Test set evaluated exactly once after all decisions are frozen</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# SECTION 5 — RUN TRAINING
# ===========================================================================
elif section == "⚙️ Run Training":
    st.markdown('<div class="section-heading">Run / Retrain ML Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <p style="font-size:14px; color:#334155; line-height:1.7; margin:0;">
            Select dataset files, validate them, then run the full training pipeline.
            All results (metrics, coefficients, curves, split info) will be saved to
            a versioned pipeline artifact paired with <code>models/model_metadata.json</code>.
            The metadata points to the matching model file so a retrain cannot mix versions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    avail_files = discover_excel_files()
    data_root = os.path.abspath(get_default_data_dir())
    default_training_files = [
        path for path in avail_files
        if os.path.dirname(os.path.abspath(path)) == data_root
    ]
    if not avail_files:
        st.error("No Excel dataset files found in the `data/` directory. Add store datasets first.")
    else:
        st.caption(
            "Only root-level demonstration workbooks are selected by default. Uploaded "
            "store workbooks are used for inference and are included in training only if "
            "you explicitly select them here."
        )
        path_labels = {fp: os.path.basename(fp) for fp in avail_files}
        selected_files = st.multiselect(
            "Select dataset files to train on",
            options=avail_files,
            default=default_training_files,
            format_func=lambda p: path_labels.get(p, os.path.basename(p)),
        )

        col_v, col_t = st.columns([1, 1])
        with col_v:
            if st.button("🔍 Validate Datasets", use_container_width=True):
                if not selected_files:
                    st.error("Select at least one dataset.")
                else:
                    with st.spinner("Validating..."):
                        pf = validate_training_sources(selected_files)

                    if pf.get("files_rejected"):
                        for r in pf["files_rejected"]:
                            st.error(f"❌ {r['file']}: {r['reason']}")
                    if pf.get("files_accepted"):
                        st.success(f"✅ {len(pf['files_accepted'])} file(s) valid: {', '.join(pf['files_accepted'])}")
                    for w in pf.get("warnings", []):
                        st.warning(f"⚠️ {w}")
                    if pf.get("is_valid"):
                        st.info(f"Combined: {pf['total_rows']:,} rows, {pf['date_min']} → {pf['date_max']}")

        with col_t:
            if st.button("🚀 Run Training Pipeline", use_container_width=True):
                if not selected_files:
                    st.error("Select at least one dataset.")
                else:
                    progress_bar = st.progress(0, text="Initialising pipeline…")
                    with st.spinner("Training — this may take 10–60 seconds…"):
                        try:
                            progress_bar.progress(20, text="Validating sources…")
                            new_meta = run_training_pipeline(store_paths=selected_files)
                            progress_bar.progress(100, text="Done!")
                            st.cache_resource.clear()
                            st.success(
                                f"✅ Training complete in **{new_meta['train_duration_seconds']}s** — "
                                f"model = {new_meta['algorithm']} · "
                                f"τ* = {new_meta['optimal_decision_threshold']} · "
                                f"Accuracy = {new_meta['test_metrics']['accuracy']*100:.2f}% · "
                                f"F1 = {new_meta['test_metrics']['f1_score']:.4f}"
                            )
                            st.balloons()
                            st.rerun()
                        except Exception as exc:
                            progress_bar.empty()
                            st.error(f"Training failed: {exc}")

    # Current model summary if available
    if meta:
        st.markdown('<div class="section-heading">Current Saved Model Summary</div>', unsafe_allow_html=True)
        tm = meta.get("test_metrics", {})
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Trained At", meta.get("training_timestamp", "—")[:10])
        sm2.metric("Test Accuracy", f"{tm.get('accuracy', 0)*100:.2f}%")
        sm3.metric("Test F1", f"{tm.get('f1_score', 0):.4f}")
        sm4.metric("Optimal τ*", str(meta.get("optimal_decision_threshold", "—")))

        with st.expander("📄 View Full model_metadata.json"):
            st.json(meta)
