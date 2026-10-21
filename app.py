"""
DemandWise — Smart Retail Demand Predictor
===========================================
Enterprise-grade, beginner-friendly retail demand prediction web application.
Built with Python, Streamlit, Pandas, Scikit-learn, Plotly, and SQLite.
"""

import os
import json
import uuid
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Package modules
from src.database import (
    init_db,
    get_stores_for_user,
    get_store_by_id,
    add_store,
    update_store,
    update_store_dataset_meta,
    delete_store,
    seed_demo_stores_for_user,
    get_business_profile,
    upsert_business_profile
)
from src.auth import (
    register_user,
    authenticate_user
)
from src.data_loader import (
    discover_excel_files,
    load_store_dataset,
    resolve_dataset_path,
    filter_dataset_by_date_range,
    load_all_demo_stores,
    get_default_data_dir,
    get_user_data_dir
)
from src.data_validation import validate_excel_file
from src.feature_engineering import (
    compute_features_and_target,
    ALL_MODEL_FEATURES,
    SURGE_THRESHOLD_MULTIPLIER
)
from src.train_model import (
    load_trained_model,
    run_training_pipeline,
    validate_training_sources,
    get_default_models_dir
)
from src.model_report import create_model_bundle
from src.predict_demand import (
    predict_category_surge,
    predict_all_categories_for_store
)
from src.utils import (
    plot_store_sales_trend,
    plot_category_sales_explorer,
    plot_category_breakdown_bar,
    plot_revenue_trend,
    plot_promotion_comparison,
    plot_confusion_matrix_chart,
    plot_roc_curve_chart,
    plot_feature_importance_chart,
    plot_chronological_split_diagram
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom SaaS Theme CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DemandWise — Smart Retail Demand Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the local account database.
init_db()

# Custom SaaS styling: clean white/neutral background, deep green accent, soft borders
st.markdown("""
<style>
    /* Global typography and layout */
    .stApp {
        background-color: #F9FAFB !important;
        color: #1F2937 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Force Light Theme Inputs & Form Controls */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04) !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #15803D !important;
        box-shadow: 0 0 0 3px rgba(21, 128, 61, 0.15) !important;
    }
    div[data-baseweb="input"] input,
    input[type="text"],
    input[type="password"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-size: 14px !important;
        caret-color: #15803D !important;
    }
    div[data-baseweb="input"] svg {
        fill: #6B7280 !important;
    }

    /* Widget Labels */
    label[data-testid="stWidgetLabel"],
    .stTextInput label,
    .stSelectbox label,
    .stDateInput label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #374151 !important;
        margin-bottom: 4px !important;
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #E5E7EB !important;
        gap: 6px !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #6B7280 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 16px !important;
        border-radius: 6px 6px 0 0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #15803D !important;
        border-bottom: 2.5px solid #15803D !important;
    }
    
    /* SaaS Metric & Card styling */
    .saas-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .kpi-container {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        text-align: left;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .kpi-subtitle {
        font-size: 12px;
        color: #9CA3AF;
        margin-top: 4px;
    }
    
    /* Badges */
    .badge-surge {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-normal {
        display: inline-block;
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .badge-watch {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    
    /* Primary buttons */
    div.stButton > button:first-child {
        background-color: #15803D !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        transition: background-color 0.15s ease-in-out, transform 0.1s ease !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #166534 !important;
        color: #FFFFFF !important;
    }

    /* ===================== SIDEBAR REDESIGN ===================== */

    /* ── Sidebar width and surface ─────────────────────────────── */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
        width: 320px !important;
        background: #0B3524 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 0 !important;
        width: 320px !important;
        min-width: 0 !important;
    }
    [data-testid="stSidebarContent"] {
        padding: 0 0 16px 0 !important;
        width: 320px !important;
        min-width: 0 !important;
    }

    [data-testid="stSidebar"] button[kind="header"],
    [data-testid="collapsedControl"] {
        color: rgba(255,255,255,0.6) !important;
    }

    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
        box-sizing: border-box !important;
    }

    /* ── Branding block ────────────────────────────────────────── */
    .sb-logo-block {
        padding: 26px 20px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.09);
        margin-bottom: 6px;
    }
    .sb-logo-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 18px;
    }
    .sb-logo-icon {
        width: 48px;
        height: 48px;
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
        flex-shrink: 0;
    }
    .sb-logo-text h3 {
        margin: 0 0 5px 0;
        font-size: 18px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: 0.075em;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .sb-logo-text span {
        font-size: 10px;
        color: rgba(255,255,255,0.62) !important;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* ── Account / business card ───────────────────────────────── */
    .sb-user-pill {
        display: flex;
        align-items: center;
        gap: 13px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 13px 14px;
        width: 100%;
        min-width: 0;
    }
    .sb-user-avatar {
        width: 36px;
        height: 36px;
        min-width: 36px;
        border-radius: 50%;
        background: #16834D;
        border: 2px solid rgba(255,255,255,0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        font-weight: 700;
        color: #FFFFFF !important;
        flex-shrink: 0;
    }
    .sb-user-info {
        min-width: 0;
        flex: 1;
        overflow: hidden;
    }
    .sb-user-info b {
        display: block;
        font-size: 13.5px;
        font-weight: 700;
        color: #FFFFFF !important;
        line-height: 1.35;
        overflow-wrap: anywhere;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        overflow: hidden;
    }
    .sb-user-info small {
        display: block;
        font-size: 11px;
        color: rgba(255,255,255,0.62) !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Nav section label ─────────────────────────────────────── */
    .sb-nav-label {
        padding: 16px 20px 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.52) !important;
    }

    /* Keep the existing radio widget and state, but present it as a nav list. */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 5px !important;
        padding: 0 12px !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        min-height: 46px !important;
        padding: 11px 14px !important;
        border-radius: 9px !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, color 0.15s ease !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: rgba(255,255,255,0.78) !important;
        margin: 0 !important;
        border: none !important;
        background: transparent !important;
        letter-spacing: 0.01em !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p {
        margin: 0 !important;
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        line-height: inherit !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
        background: rgba(255,255,255,0.07) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
        background: rgba(74,222,128,0.14) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: inset 3px 0 0 #4ADE80 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {
        display: none !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 8px 16px !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {
        color: rgba(255,255,255,0.60) !important;
    }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
    }

    .sb-footer {
        padding: 15px 20px 18px;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin-top: 16px;
    }
    .sb-footer span {
        font-size: 11px;
        color: rgba(255,255,255,0.48) !important;
    }

    /* ── Responsive sizing: the sidebar remains collapsible ───── */
    @media (max-width: 1100px) {
        [data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 280px !important;
            width: 280px !important;
        }
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarContent"] {
            width: 280px !important;
        }
    }
    @media (max-width: 760px) {
        [data-testid="stSidebar"] {
            min-width: 240px !important;
            max-width: 240px !important;
            width: 240px !important;
        }
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarContent"] {
            width: 240px !important;
        }
        .sb-logo-block {
            padding: 22px 16px 17px;
        }
        .sb-nav-label {
            padding-left: 16px;
        }
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        padding: 2rem 2rem 3rem !important;
    }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.5rem 1.25rem 2.5rem !important;
        }
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 2rem !important;
        }
    }
    /* ============================================================ */
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Currency helpers
# -----------------------------------------------------------------------------
CURRENCY_SYMBOLS = {
    "USD ($)": "$",
    "EUR (€)": "€",
    "GBP (£)": "£",
    "INR (₹)": "₹",
    "CAD ($)": "CA$",
}

def get_currency_symbol(currency_str: str) -> str:
    """Returns the symbol character for a currency preference string."""
    return CURRENCY_SYMBOLS.get(currency_str, "$")

def fmt_currency(value: float, symbol: str) -> str:
    """Formats a numeric value with the provided currency symbol."""
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:.1f}K"
    return f"{symbol}{value:,.0f}"


# -----------------------------------------------------------------------------
# 3. Caching Helpers for Performance & Memory Efficiency
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_model(models_dir: str, metadata_mtime_ns: int, metadata_size: int):
    """Loads and caches the fitted pipeline, invalidating cache when metadata changes."""
    return load_trained_model(models_dir)


@st.cache_data(show_spinner=False)
def _load_store_dataframe(file_path: str, store_name: str, modified_ns: int, file_size: int):
    """Load a configured store workbook; stat values invalidate stale cache entries."""
    return load_store_dataset(file_path, store_name)


def get_store_dataset(store: dict):
    """Load a store dataset and return explicit errors instead of empty-data fallbacks."""
    file_path = store.get('dataset_path')
    if not file_path:
        return None, None, f"No dataset is connected to {store['store_name']}."

    resolved_path = resolve_dataset_path(file_path)
    try:
        stat = os.stat(resolved_path)
    except OSError as exc:
        return None, None, f"Dataset file is unavailable at '{resolved_path}': {exc}"

    try:
        dataset, metadata = _load_store_dataframe(
            resolved_path,
            store['store_name'],
            stat.st_mtime_ns,
            stat.st_size
        )
    except (OSError, ValueError) as exc:
        return None, None, str(exc)
    return dataset, metadata, None


@st.cache_data(show_spinner=False)
def load_all_combined_stores_data():
    """Loads all store data for multi-store views."""
    cal, metas = load_all_demo_stores()
    return cal, metas


# -----------------------------------------------------------------------------
# 4. Session State Initialization
# -----------------------------------------------------------------------------
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'business_name' not in st.session_state:
    st.session_state['business_name'] = "DemandWise Retail"
if 'owner_name' not in st.session_state:
    st.session_state['owner_name'] = "Store Owner"
if 'currency' not in st.session_state:
    st.session_state['currency'] = "USD ($)"
if 'currency_symbol' not in st.session_state:
    st.session_state['currency_symbol'] = "$"


# -----------------------------------------------------------------------------
# 5. Authentication View (When Not Signed In)
# -----------------------------------------------------------------------------
def render_auth_view():
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; margin-bottom: 24px;">
        <span style="font-size: 40px;">📦</span>
        <h1 style="margin: 6px 0 2px 0; font-size: 26px; color: #111827;">DemandWise</h1>
        <p style="color: #6B7280; font-size: 14px; margin: 0;">Smart Retail Demand Predictor for Multi-Store Retail</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        with st.container(border=True):
            tab_login = st.container()
            tab_signup = st.container()

            with tab_login:
                st.subheader("Sign In")
                st.markdown("<p style='font-size:13px; color:#4B5563; margin: 4px 0 14px 0;'>Sign in to access your retail stores and demand predictions.</p>", unsafe_allow_html=True)
                
                login_user = st.text_input("Username", key="login_username")
                login_pwd = st.text_input("Password", type="password", key="login_password")

                if st.button("Sign In to DemandWise", use_container_width=True):
                    ok, msg, u, b = authenticate_user(login_user, login_pwd)
                    if ok:
                        st.session_state['authenticated'] = True
                        st.session_state['user_id'] = u['id']
                        st.session_state['username'] = u['username']
                        if b:
                            st.session_state['business_name'] = b['business_name']
                            st.session_state['owner_name'] = b['owner_name']
                            st.session_state['currency'] = b.get('currency', 'USD ($)')
                            st.session_state['currency_symbol'] = get_currency_symbol(
                                b.get('currency', 'USD ($)')
                            )
                        st.success("Welcome back! Loading your stores...")
                        st.rerun()
                    else:
                        st.error(msg)

            with tab_signup:
                st.divider()
                st.subheader("Create New Account")
                st.markdown("<p style='font-size:13px; color:#4B5563; margin: 4px 0 14px 0;'>Register a new business account and associate your store workbooks.</p>", unsafe_allow_html=True)
                su_user = st.text_input("Choose Username", key="su_username")
                su_email = st.text_input("Email Address", key="su_email")
                su_biz = st.text_input("Business / Chain Name", placeholder="e.g. Apex Supermarkets", key="su_biz")
                su_owner = st.text_input("Owner / Manager Name", placeholder="e.g. Sarah Jenkins", key="su_owner")
                su_p1 = st.text_input("Password (min 6 characters)", type="password", key="su_p1")
                su_p2 = st.text_input("Confirm Password", type="password", key="su_p2")

                seed_opt = st.checkbox(
                    "Load sample demonstration stores on signup",
                    value=False,
                    key="su_seed_demo"
                )
                if seed_opt:
                    st.info(
                        "ℹ️ **About demonstration stores:** These 5 stores use shared sample "
                        "Excel files from the `data/` folder. All users who enable this option "
                        "will see the **same sample data** — it is not unique to your account. "
                        "To work with your own data, leave this unchecked and upload your own "
                        "Excel files in **My Stores** after signing in."
                    )
                else:
                    st.caption(
                        "✅ You will start with a clean, empty account. "
                        "Add your own stores and upload your sales data in **My Stores**."
                    )

                if st.button("Create Account", use_container_width=True):
                    ok, msg, new_user = register_user(
                        username=su_user,
                        email=su_email,
                        password=su_p1,
                        confirm_password=su_p2,
                        business_name=su_biz,
                        owner_name=su_owner,
                        seed_demo=seed_opt
                    )
                    if ok:
                        if seed_opt:
                            st.success(f"{msg} Demonstration stores loaded. You can now sign in above.")
                        else:
                            st.success(f"{msg} Your account is ready. Sign in above and add your stores in My Stores.")
                    else:
                        st.error(msg)


# -----------------------------------------------------------------------------
# 6. Main Application View (Authenticated Session)
# -----------------------------------------------------------------------------
def render_main_app():
    user_id = st.session_state['user_id']
    username = st.session_state['username']
    biz_name = st.session_state.get('business_name', 'Retail Business')
    owner_name = st.session_state.get('owner_name', 'Store Owner')
    currency_sym = st.session_state.get('currency_symbol', '$')

    # Sidebar Header & Navigation
    with st.sidebar:
        # ── Logo + User pill ──────────────────────────────────────
        avatar_letter = (owner_name[0] if owner_name else "U").upper()
        safe_biz = biz_name if biz_name else "My Business"
        safe_owner = owner_name if owner_name else "Account"
        safe_username = username if username else ""
        account_sub = f"@{safe_username}" if safe_username else safe_owner
        st.markdown(f"""
        <div class="sb-logo-block">
            <div class="sb-logo-row">
                <div class="sb-logo-icon">📦</div>
                <div class="sb-logo-text">
                    <h3>DemandWise</h3>
                    <span>Retail Intelligence</span>
                </div>
            </div>
            <div class="sb-user-pill">
                <div class="sb-user-avatar">{avatar_letter}</div>
                <div class="sb-user-info">
                    <b>{safe_biz}</b>
                    <small>{account_sub}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Nav section label ─────────────────────────────────────
        st.markdown('<div class="sb-nav-label">Main Menu</div>', unsafe_allow_html=True)

        # ── Navigation radio (styled via CSS) ─────────────────────
        page = st.radio(
            "Navigation",
            [
                "🏠 Overview",
                "🏪 My Stores",
                "📈 Explore Sales",
                "⚡ Check Demand",
                "👤 Account / Sign Out"
            ],
            label_visibility="collapsed"
        )

        # ── Footer ────────────────────────────────────────────────
        st.markdown("""
        <div class="sb-footer">
            <span>DemandWise &nbsp;·&nbsp; v1.0</span>
        </div>
        """, unsafe_allow_html=True)

    # Load User Stores from SQLite
    stores = get_stores_for_user(user_id)
    
    # Pre-load model
    models_dir = get_default_models_dir()
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    try:
        metadata_stat = os.stat(metadata_path)
    except FileNotFoundError:
        metadata_mtime_ns = 0
        metadata_size = 0
    else:
        metadata_mtime_ns = metadata_stat.st_mtime_ns
        metadata_size = metadata_stat.st_size
    pipeline, model_meta = get_cached_model(
        models_dir,
        metadata_mtime_ns,
        metadata_size
    )

    # -------------------------------------------------------------------------
    # PAGE 1: OVERVIEW ("Your Store at a Glance")
    # -------------------------------------------------------------------------
    if page == "🏠 Overview":
        st.title("Overview")

        if not stores:
            st.markdown("""
            <div style="text-align:center; padding: 48px 24px;">
                <div style="font-size:48px; margin-bottom:12px;">🏪</div>
                <h3 style="color:#111827; margin-bottom:8px;">No stores yet</h3>
                <p style="color:#6B7280; font-size:15px; max-width:420px; margin:0 auto 20px auto;">
                    Go to <b>My Stores</b> to add your first retail location and upload your sales data.
                    Once a dataset is connected, predictions and charts will appear here.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return

        # Store selector
        store_names = [s['store_name'] for s in stores]
        sel_store = st.selectbox("Choose store", store_names, index=0, key="ov_store_sel")
        curr_store = next((s for s in stores if s['store_name'] == sel_store), None)
        
        # Load dataset for selected store
        cal_df = None
        dataset_meta = None
        dataset_error = None
        if curr_store:
            cal_df, dataset_meta, dataset_error = get_store_dataset(curr_store)
        if dataset_error:
            st.error(f"Dataset unavailable for {sel_store}: {dataset_error}")

        # ---- Reporting period (90-day rolling window) ----
        OVERVIEW_DAYS = 90
        period_df = None
        period_label = f"Last {OVERVIEW_DAYS} days"
        if cal_df is not None and not cal_df.empty:
            max_date = cal_df['date'].max()
            period_start = max_date - pd.Timedelta(days=OVERVIEW_DAYS - 1)
            period_df = filter_dataset_by_date_range(cal_df, period_start, max_date)
            period_df = period_df[period_df['store_name'] == sel_store].copy()
            period_label = (
                f"{period_start.strftime('%b %d, %Y')} – {max_date.strftime('%b %d, %Y')}"
            )

        # ---- KPI Cards ----
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("My Stores", len(stores))

        if period_df is not None and not period_df.empty:
            total_units = period_df['sales'].sum()
            total_rev = period_df['revenue'].sum()
            latest_date = cal_df['date'].max()
            k2.metric("Units Sold", f"{total_units:,.0f}", help=f"Period: {period_label}")
            k3.metric(
                "Sales Revenue",
                fmt_currency(total_rev, currency_sym),
                help=f"Period: {period_label}"
            )
            k4.metric("Latest Data", latest_date.strftime("%b %d, %Y"))
        else:
            k2.metric("Units Sold", "—")
            k3.metric("Sales Revenue", "—")
            k4.metric("Latest Data", "—")

        st.caption(f"📅 Sales metrics cover: **{period_label}** (90-day rolling window for {sel_store})")

        # ---- Main Columns: Categories to Watch & Recent Trend ----
        left_col, right_col = st.columns([1.2, 1])

        with left_col:
            st.markdown("### Categories to Review")

            if cal_df is not None and pipeline is not None and model_meta is not None:
                with st.spinner("Generating demand outlook..."):
                    preds = predict_all_categories_for_store(
                        cal_df, sel_store, pipeline=pipeline, metadata=model_meta
                    )

                if preds:
                    observed_surge_count = sum(1 for p in preds if p['observed_surge'])
                    surge_count = sum(1 for p in preds if p['badge_status'] == 'SURGE')
                    watch_count = sum(1 for p in preds if p['badge_status'] == 'WATCH')
                    total_cats = len(preds)

                    if observed_surge_count > 0:
                        st.error(
                            f"⚡ Recent sales surged in {observed_surge_count} of "
                            f"{total_cats} categories (at least 30% above the previous week)."
                        )
                    elif all(p['observed_surge_available'] for p in preds):
                        st.success("✅ No 30%+ increase was observed in the latest complete 7-day period.")
                    else:
                        st.info("Recent surge status is unavailable until 14 consecutive days of data are present.")

                    if surge_count > 0:
                        st.error(f"Next-week model forecast: potential surge in {surge_count} categories.")
                    elif watch_count > 0:
                        st.warning(
                            f"Next-week model forecast: {watch_count} categories are near the surge threshold — monitor closely."
                        )

                    for p in preds:
                        with st.container(border=True):
                            item_col, status_col = st.columns([2, 1])
                            item_col.markdown(f"**{p['category']}**")
                            item_col.caption(
                                f"Latest: {p['feature_values']['sales_today']:.0f} units · "
                                f"7-day avg: {p['feature_values']['past_7d_avg']:.1f} units/day"
                            )
                            if p['observed_surge']:
                                status_col.error("⚡ Recent Surge")
                                item_col.caption(
                                    f"Observed 7-day average: {p['observed_recent_avg']:.1f} units/day "
                                    f"({p['observed_change_pct']:.0f}% above the previous week)."
                                )
                            elif p["badge_status"] == "SURGE":
                                status_col.error("⚡ Forecast Surge")
                            elif p["badge_status"] == "WATCH":
                                status_col.warning("👁️ Forecast: Monitor")
                            else:
                                status_col.success("✓ No Recent Surge")
                            st.caption(
                                f"Next 7-day model forecast: {p['status_label']} "
                                f"({p['probability_pct']:.1f}% score)."
                            )
                else:
                    st.info("Not enough sales history yet for demand predictions.")
            elif dataset_error:
                st.info("Category analysis is unavailable until the dataset issue is resolved.")
            elif pipeline is None:
                st.warning(
                    "Demand outlook requires a trained model. "
                    "Go to **📦 Model Insights → Train / Retrain Model** to train the pipeline."
                )
            else:
                st.info(f"Upload sales data for {sel_store} in My Stores to enable predictions.")

        with right_col:
            st.markdown(f"### Recent Sales — Last 45 Days")
            if cal_df is not None and not cal_df.empty:
                fig_trend = plot_store_sales_trend(cal_df, sel_store, days=45)
                st.plotly_chart(fig_trend, use_container_width=True)
            elif dataset_error:
                st.info("The sales chart is unavailable until the dataset issue is resolved.")
            else:
                st.info("Upload sales data to see this chart.")

    # -------------------------------------------------------------------------
    # PAGE 2: MY STORES (Store Management, Upload & Validation)
    # -------------------------------------------------------------------------
    elif page == "🏪 My Stores":
        st.title("Store Management & Datasets")
        st.markdown("<p style='color:#6B7280; margin-top:-8px;'>Manage retail locations and link their daily Excel sales datasets.</p>", unsafe_allow_html=True)

        t_list, t_add = st.tabs(["📋 Your Stores", "➕ Add New Store"])

        with t_list:
            # ── Detect legacy shared demo stores ──────────────────────────────
            default_data_dir = get_default_data_dir()
            demo_store_ids = [
                s['id'] for s in stores
                if s.get('is_demo') == 1 or (
                    s.get('dataset_path') and
                    os.path.dirname(os.path.abspath(s['dataset_path'])) == os.path.abspath(default_data_dir)
                )
            ]

            if not stores:
                # ── Empty-state onboarding ─────────────────────────────────────
                st.markdown("""
                <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:14px;
                            padding:28px 28px; margin-bottom:20px;">
                    <h3 style="color:#14532D; margin:0 0 8px 0;">🏪 Add Your First Store</h3>
                    <p style="color:#166534; font-size:14px; margin:0 0 16px 0;">
                        Your account is clean and ready. Here's how to get started:
                    </p>
                    <ol style="color:#15803D; font-size:14px; margin:0; padding-left:20px; line-height:2;">
                        <li>Click <b>"Add New Store"</b> tab above to register your retail location.</li>
                        <li>Open the store card and use <b>"Upload Excel Dataset"</b> to connect your sales file.</li>
                        <li>Once uploaded, go to <b>Overview</b> and <b>Check Demand</b> to see predictions.</li>
                    </ol>
                    <p style="color:#6B7280; font-size:12px; margin:16px 0 0 0;">
                        Your Excel file must have columns: <code>date, store_name, category, product, sales, unit_price</code>.
                        A <code>promotion</code> column is optional.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ── Cleanup action for legacy shared demo stores ──────────────
                if demo_store_ids:
                    if st.button(
                        f"🗑️ Remove {len(demo_store_ids)} shared demo store(s) from my account",
                        key="remove_demo_stores_btn"
                    ):
                        for sid in demo_store_ids:
                            delete_store(sid, user_id)
                        st.cache_data.clear()
                        st.success("Shared demo stores removed. Add your own stores using the 'Add New Store' tab.")
                        st.rerun()
                for s in stores:
                    with st.expander(
                        f"📍 {s['store_name']}  ({s.get('store_code') or 'Store'})",
                        expanded=False
                    ):
                        sc1, sc2 = st.columns([1.5, 1])
                        with sc1:
                            st.write(f"**Type:** {s.get('store_type', 'Retail Branch')}")
                            st.write(f"**Location:** {s.get('location', 'N/A')}")
                            st.markdown("#### Dataset Status")

                            fpath = s.get('dataset_path')
                            fname = s.get('dataset_filename') or (
                                os.path.basename(fpath) if fpath else None
                            )
                            dataset_df, dataset_meta, dataset_error = get_store_dataset(s)
                            if dataset_df is not None:
                                st.success("✅ Dataset connected and ready.")
                                if fname:
                                    st.write(f"**File:** `{fname}`")
                                if dataset_meta.get('date_min') and dataset_meta.get('date_max'):
                                    st.write(
                                        f"**Date Range:** {dataset_meta['date_min']} "
                                        f"to {dataset_meta['date_max']}"
                                    )
                                st.write(f"**Records:** {dataset_meta['row_count']:,}")
                                if s.get('dataset_upload_date'):
                                    st.write(f"**Last Imported:** {s['dataset_upload_date']}")
                                if s.get('dataset_revenue_calculated'):
                                    st.caption(
                                        "ℹ️ Revenue in this dataset was calculated from "
                                        "sales × unit_price (not directly recorded)."
                                    )
                                if s.get('dataset_column_warnings'):
                                    st.caption(f"Column mappings applied: {s['dataset_column_warnings']}")
                            elif fpath:
                                st.error(f"Dataset is not connected: {dataset_error}")
                            else:
                                st.info(dataset_error)

                        with sc2:
                            st.markdown("#### Upload / Replace Dataset")
                            up_file = st.file_uploader(
                                f"Upload Excel for {s['store_name']}",
                                type=['xlsx'],
                                key=f"up_{s['id']}"
                            )
                            if up_file is not None:
                                file_bytes = up_file.getvalue()
                                val_report = validate_excel_file(file_bytes)

                                if val_report['is_valid']:
                                    st.success(
                                        f"✅ Validation passed — "
                                        f"{val_report['row_count']:,} rows, "
                                        f"{val_report['unique_days']} days "
                                        f"({val_report['date_min']} to {val_report['date_max']})"
                                    )
                                    # Show any warnings (aliases, calculated revenue, duplicates)
                                    for w in val_report.get('warnings', []):
                                        st.warning(f"⚠️ {w}")

                                    # Preview
                                    st.dataframe(
                                        val_report['cleaned_df'].head(5),
                                        use_container_width=True
                                    )

                                    if st.button(
                                        f"Save & Connect to {s['store_name']}",
                                        key=f"save_up_{s['id']}"
                                    ):
                                        save_dir = get_user_data_dir(user_id)
                                        os.makedirs(save_dir, exist_ok=True)
                                        upload_filename = os.path.basename(up_file.name)
                                        safe_name = (
                                            f"User_Store_{s['id']}_{uuid.uuid4().hex}_"
                                            f"{upload_filename}"
                                        )
                                        save_path = os.path.join(save_dir, safe_name)
                                        save_file_created = False

                                        try:
                                            with open(save_path, "xb") as f_out:
                                                save_file_created = True
                                                f_out.write(file_bytes)

                                            # Build warning summary for DB storage
                                            alias_warnings = [
                                                w for w in val_report.get('warnings', [])
                                                if 'mapped to' in w
                                            ]
                                            col_warn_str = (
                                                "; ".join(alias_warnings)
                                                if alias_warnings else None
                                            )

                                            updated = update_store_dataset_meta(
                                                store_id=s['id'],
                                                user_id=user_id,
                                                dataset_path=save_path,
                                                dataset_filename=up_file.name,
                                                dataset_upload_date=datetime.datetime.now().strftime(
                                                    "%Y-%m-%d %H:%M"
                                                ),
                                                dataset_row_count=val_report['row_count'],
                                                dataset_date_min=val_report.get('date_min'),
                                                dataset_date_max=val_report.get('date_max'),
                                                dataset_revenue_calculated=int(
                                                    val_report.get('revenue_calculated', False)
                                                ),
                                                dataset_column_warnings=col_warn_str
                                            )

                                            if updated:
                                                st.cache_data.clear()
                                                st.success(
                                                    f"✅ Dataset updated successfully for {s['store_name']}!"
                                                )
                                                st.rerun()
                                            else:
                                                if os.path.exists(save_path):
                                                    os.remove(save_path)
                                                st.error(
                                                    "Dataset save failed: could not update store record. "
                                                    "Previous dataset has been preserved."
                                                )
                                        except Exception as exc:
                                            if save_file_created and os.path.exists(save_path):
                                                try:
                                                    os.remove(save_path)
                                                except OSError as cleanup_exc:
                                                    st.error(
                                                        f"Could not remove the incomplete upload "
                                                        f"'{save_path}': {cleanup_exc}"
                                                    )
                                            st.error(
                                                f"Upload failed: {str(exc)}. "
                                                "Previous dataset has been preserved."
                                            )
                                else:
                                    st.error("❌ Validation failed — dataset not saved:")
                                    for err in val_report['errors']:
                                        st.write(f"• {err}")
                                    if val_report.get('warnings'):
                                        for w in val_report['warnings']:
                                            st.caption(f"Note: {w}")

                        # ---- Delete Store (with confirmation) ----
                        st.markdown("---")
                        del_key = f"confirm_del_{s['id']}"
                        if del_key not in st.session_state:
                            st.session_state[del_key] = False

                        if not st.session_state[del_key]:
                            if st.button(
                                f"🗑️ Delete {s['store_name']}",
                                key=f"del_btn_{s['id']}"
                            ):
                                st.session_state[del_key] = True
                                st.rerun()
                        else:
                            st.warning(
                                f"⚠️ Confirm: permanently delete **{s['store_name']}** "
                                "and remove it from your account? "
                                "This does not delete the dataset file on disk."
                            )
                            dc1, dc2 = st.columns(2)
                            with dc1:
                                if st.button("Yes, delete store", key=f"del_confirm_{s['id']}"):
                                    delete_store(s['id'], user_id)
                                    st.cache_data.clear()
                                    st.session_state.pop(del_key, None)
                                    st.warning(f"Store '{s['store_name']}' deleted.")
                                    st.rerun()
                            with dc2:
                                if st.button("Cancel", key=f"del_cancel_{s['id']}"):
                                    st.session_state[del_key] = False
                                    st.rerun()

        with t_add:
            st.markdown("### Add a New Retail Location")
            st.caption(
                "After adding a store, open its card in the **Your Stores** tab to upload your Excel dataset."
            )
            with st.form("add_store_form"):
                n_name = st.text_input("Store Name", placeholder="e.g. Main Branch")
                n_code = st.text_input("Store Identifier Code", placeholder="e.g. STR-01")
                n_type = st.selectbox(
                    "Store Type",
                    ["Flagship Downtown", "Urban Supermarket", "Neighborhood Store",
                     "Suburban Retail", "Shopping Mall Outlet", "Other"]
                )
                n_loc = st.text_input("Location / Address", placeholder="e.g. 104 Main Street")

                submitted = st.form_submit_button("Add Store")
                if submitted:
                    if not n_name.strip():
                        st.error("Store name is required.")
                    else:
                        add_store(
                            user_id=user_id,
                            store_name=n_name.strip(),
                            store_code=n_code.strip() or None,
                            store_type=n_type,
                            location=n_loc.strip() or None,
                            dataset_path=None,
                            is_demo=0
                        )
                        st.success(
                            f"✅ Store **'{n_name}'** added! Open it in 'Your Stores' to upload your dataset."
                        )
                        st.rerun()

    # -------------------------------------------------------------------------
    # PAGE 3: EXPLORE SALES (Trends, Revenue, Promotions)
    # -------------------------------------------------------------------------
    elif page == "📈 Explore Sales":
        st.title("Explore Past Sales & Trends")
        st.markdown("<p style='color:#6B7280; margin-top:-8px;'>Interactive analysis of historical category sales, revenue, and promotion patterns.</p>", unsafe_allow_html=True)

        if not stores:
            st.info("No stores available. Add a store in **My Stores** to explore sales.")
            return

        # Filter Bar
        f1, f2, f3 = st.columns([1, 1, 1.2])
        store_names = [s['store_name'] for s in stores]
        with f1:
            sel_store = st.selectbox("Store", store_names, key="exp_store")
        
        curr_store = next((s for s in stores if s['store_name'] == sel_store), None)
        cal_df = None
        dataset_meta = None
        dataset_error = None
        if curr_store:
            cal_df, dataset_meta, dataset_error = get_store_dataset(curr_store)

        if cal_df is None or cal_df.empty:
            if dataset_error:
                st.error(f"Dataset unavailable for {sel_store}: {dataset_error}")
            else:
                st.info(f"No sales records are available for {sel_store}. Upload an Excel workbook in My Stores.")
            return

        categories = sorted(cal_df['category'].unique().tolist())
        with f2:
            sel_cat = st.selectbox("Category", ["All Categories"] + categories, key="exp_cat")

        with f3:
            min_d = cal_df['date'].min().to_pydatetime()
            max_d = cal_df['date'].max().to_pydatetime()
            date_range = st.date_input("Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)

        # Slice data
        filtered_df = cal_df.copy()
        if len(date_range) == 2:
            filtered_df = filter_dataset_by_date_range(
                filtered_df, date_range[0], date_range[1]
            )
        filtered_df = filtered_df[filtered_df['store_name'] == sel_store].copy()
        if sel_cat != "All Categories":
            filtered_df = filtered_df[filtered_df['category'] == sel_cat]

        if filtered_df.empty:
            st.warning("No records match the selected filters.")
            return

        # Period label
        if len(date_range) == 2:
            period_str = (
                f"{pd.to_datetime(date_range[0]).strftime('%b %d, %Y')} – "
                f"{pd.to_datetime(date_range[1]).strftime('%b %d, %Y')}"
            )
        else:
            period_str = "Selected period"
        st.caption(f"📅 Metrics and charts cover: **{period_str}**")

        # Summary KPIs for the slice
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Units Sold</div>
                <div class="kpi-value">{filtered_df['sales'].sum():,.0f}</div>
                <div class="kpi-subtitle">Total in selected period</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k2:
            rev_total = filtered_df['revenue'].sum()
            rev_note = ""
            if curr_store and curr_store.get('dataset_revenue_calculated'):
                rev_note = "Calculated"
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Revenue {f'({rev_note})' if rev_note else ''}</div>
                <div class="kpi-value">{fmt_currency(rev_total, currency_sym)}</div>
                <div class="kpi-subtitle">Period total</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            avg_daily = filtered_df.groupby('date')['sales'].sum().mean()
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Daily Average</div>
                <div class="kpi-value">{avg_daily:,.1f}</div>
                <div class="kpi-subtitle">Units per day</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            promo_days = int((filtered_df['promotion'] == 1).sum())
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Promotion Days</div>
                <div class="kpi-value">{promo_days:,}</div>
                <div class="kpi-subtitle">Records with promo flag</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Charts
        c_left, c_right = st.columns(2)
        with c_left:
            if sel_cat != "All Categories":
                fig_cat = plot_category_sales_explorer(
                    cal_df, sel_store, sel_cat,
                    start_date=date_range[0] if len(date_range) == 2 else None,
                    end_date=date_range[1] if len(date_range) == 2 else None
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                fig_bar = plot_category_breakdown_bar(filtered_df, sel_store)
                st.plotly_chart(fig_bar, use_container_width=True)

        with c_right:
            fig_rev = plot_revenue_trend(filtered_df, sel_store)
            st.plotly_chart(fig_rev, use_container_width=True)

        # Promotion comparison — only when there is non-trivial promotion data
        has_promo_variation = (
            filtered_df['promotion'].nunique() > 1 and
            filtered_df['promotion'].sum() >= 3
        )
        if has_promo_variation:
            st.markdown("### 🏷️ Promotion Comparison")
            st.caption(
                "This chart compares average daily sales on promotional vs. non-promotional days. "
                "It shows correlation only — it does not prove that promotions caused any change in sales."
            )
            fig_promo = plot_promotion_comparison(
                filtered_df, sel_store,
                category=sel_cat if sel_cat != "All Categories" else None
            )
            st.plotly_chart(fig_promo, use_container_width=True)
        else:
            st.info(
                "Promotion comparison is not shown: insufficient promotional day records "
                "in the selected filters to produce a meaningful comparison."
            )

        # Underlying Data Table
        with st.expander("📄 View Underlying Daily Records", expanded=False):
            st.dataframe(filtered_df.sort_values('date', ascending=False), use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 4: CHECK DEMAND (The Core Prediction Workflow)
    # -------------------------------------------------------------------------
    elif page == "⚡ Check Demand":
        st.title("Check Demand Surge")
        st.markdown("<p style='color:#6B7280; margin-top:-8px;'>Assess whether a product category may experience a demand surge over the next 7 days.</p>", unsafe_allow_html=True)

        if not stores:
            st.info("No stores configured. Add a store in **My Stores** to generate predictions.")
            return

        # 3-Step Selection Card
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        s1, s2, s3 = st.columns([1.2, 1.2, 1])

        store_names = [s['store_name'] for s in stores]
        with s1:
            st.markdown("<b style='font-size:13px; color:#15803D;'>STEP 1</b>", unsafe_allow_html=True)
            pred_store = st.selectbox("Choose Store", store_names, key="pred_store_sel")

        curr_store = next((s for s in stores if s['store_name'] == pred_store), None)
        cal_df = None
        dataset_error = None
        if curr_store:
            cal_df, _, dataset_error = get_store_dataset(curr_store)

        if cal_df is None or cal_df.empty:
            st.markdown('</div>', unsafe_allow_html=True)
            if dataset_error:
                st.error(f"Dataset unavailable for {pred_store}: {dataset_error}")
            else:
                st.info(f"No sales records are available for {pred_store}. Upload a dataset in My Stores.")
            return

        store_data = cal_df[cal_df['store_name'] == pred_store].copy()
        cats = sorted(store_data['category'].dropna().unique().tolist())
        if not cats:
            st.markdown('</div>', unsafe_allow_html=True)
            st.info(f"No product categories were found in the dataset for {pred_store}.")
            return
        with s2:
            st.markdown("<b style='font-size:13px; color:#15803D;'>STEP 2</b>", unsafe_allow_html=True)
            pred_cat = st.selectbox("Choose Product Category", cats, key="pred_cat_sel")

        with s3:
            st.markdown("<b style='font-size:13px; color:#15803D;'>STEP 3</b>", unsafe_allow_html=True)
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            check_clicked = st.button("⚡ Check Demand", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if pipeline is None or model_meta is None:
            st.warning(
                "🔧 **No trained model found.** "
                "Train a model before requesting demand predictions. "
                "The selected store's available categories are shown above."
            )
            return

        if not check_clicked:
            return

        # ---- Stale-data warning ----
        category_data = store_data[store_data['category'] == pred_cat]
        if category_data['date'].nunique() < 31:
            st.info(
                f"Insufficient history for {pred_cat} at {pred_store}: "
                "at least 31 calendar days are required to calculate every feature used by the model."
            )
            return

        latest_obs = store_data['date'].max()
        days_old = (pd.Timestamp.now().normalize() - latest_obs).days
        if days_old > 14:
            st.warning(
                f"⚠️ **Stale data warning:** The most recent observation for "
                f"**{pred_store}** is from {latest_obs.strftime('%b %d, %Y')} "
                f"({days_old} days ago). Predictions are based on historical data "
                f"from that date and may not reflect current conditions."
            )

        # Generate Prediction
        prediction_result = predict_category_surge(
            cal_df,
            pred_store,
            pred_cat,
            pipeline=pipeline,
            metadata=model_meta
        )

        if prediction_result.get('success'):
            res = prediction_result
            if res['observed_surge']:
                st.error(
                    f"⚡ **Recent sales surge detected:** the latest 7-day average "
                    f"({res['observed_recent_avg']:.1f} units/day) is "
                    f"{res['observed_change_pct']:.0f}% above the previous 7 days "
                    f"({res['observed_previous_avg']:.1f} units/day), from "
                    f"{res['observed_window_start']} to {res['observed_window_end']}."
                )
            elif res['observed_surge_available']:
                st.success(
                    "No 30%+ increase was observed in the latest complete 7-day period."
                )
            else:
                st.info(
                    "Recent surge status is unavailable because the dataset does not contain "
                    "14 consecutive days of sales."
                )
            
            # Prominent Result Card
            badge_class = f"badge-{res['badge_status'].lower()}"

            # Map badge status to readable label
            status_labels = {
                'SURGE': 'Potential Demand Surge',
                'WATCH': 'Monitor Closely',
                'NORMAL': 'No Surge Detected',
            }
            status_text = status_labels.get(res['badge_status'], res['status_label'])
            prob_pct = res['probability_pct']
            
            st.html(f"""
            <div class="saas-card" style="border-left: 6px solid {res['badge_color']};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div style="font-size: 13px; font-weight: 600; color: #6B7280; text-transform: uppercase;">
                            Next 7-Day Model Forecast &bull; {res['store_name']}
                        </div>
                        <h2 style="margin: 4px 0 8px 0; font-size: 26px; color: #111827;">{res['category']}</h2>
                        <span class="{badge_class}" style="font-size: 15px; padding: 6px 16px;">
                            {status_text}
                        </span>
                    </div>
                    <div style="text-align: right; background-color: #F9FAFB; padding: 12px 18px; border-radius: 10px; border: 1px solid #E5E7EB;">
                        <div style="font-size: 11px; color: #6B7280; font-weight: 600; text-transform: uppercase;">Estimated Surge Probability</div>
                        <div style="font-size: 32px; font-weight: 800; color: {res['badge_color']}; line-height: 1.1;">{prob_pct}%</div>
                        <div style="font-size: 10px; color: #9CA3AF; margin-top: 2px;">Model probability estimate</div>
                    </div>
                </div>
                
                <hr style="border: 0; border-top: 1px solid #E5E7EB; margin: 18px 0;">
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 16px;">
                    <div>
                        <div style="font-size: 11px; color: #6B7280; font-weight: 600;">FORECAST PERIOD</div>
                        <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">
                            {res['forecast_horizon_label']}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #6B7280; font-weight: 600;">BASED ON DATA UP TO</div>
                        <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">
                            {res['evaluation_date']}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #6B7280; font-weight: 600;">LATEST DAILY SALES</div>
                        <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">
                            {res['feature_values']['sales_today']:.0f} units
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 11px; color: #6B7280; font-weight: 600;">7-DAY AVERAGE</div>
                        <div style="font-size: 14px; font-weight: 700; color: #111827; margin-top: 2px;">
                            {res['feature_values']['past_7d_avg']:.1f} units/day
                        </div>
                    </div>
                </div>
            </div>
            """)

            # ---- Explanation based on historical features ----
            if res.get('explanation'):
                st.markdown("#### What the data shows")
                st.markdown(f"> {res['explanation']}")

            # ---- Suggested Action ----
            with st.container(border=True):
                st.markdown("**📋 Suggested Next Step**")
                st.markdown(res['advisory'])

            # ---- Recent sales chart (last 60 days) ----
            st.markdown("#### Recent Sales History (Last 60 Days)")
            store_cat_data = cal_df[
                (cal_df['store_name'] == pred_store) &
                (cal_df['category'] == pred_cat)
            ].copy()
            if not store_cat_data.empty:
                chart_end = store_cat_data['date'].max()
                chart_start = chart_end - pd.Timedelta(days=59)
                fig_hist = plot_category_sales_explorer(
                    cal_df, pred_store, pred_cat,
                    start_date=chart_start,
                    end_date=chart_end
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        else:
            st.error(prediction_result.get('error', 'Prediction could not be computed.'))

    # -------------------------------------------------------------------------
    # PAGE 5: ACCOUNT / SIGN OUT
    # -------------------------------------------------------------------------
    elif page == "👤 Account / Sign Out":
        st.title("Business Profile & Account")

        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("### Business Profile")
        with st.form("biz_profile_form"):
            b_name = st.text_input("Business / Chain Name", value=biz_name)
            o_name = st.text_input("Owner Name", value=owner_name)
            _currency_opts = ["USD ($)", "EUR (€)", "GBP (£)", "INR (₹)", "CAD ($)"]
            _saved_currency = st.session_state.get('currency', 'USD ($)')
            _currency_idx = _currency_opts.index(_saved_currency) if _saved_currency in _currency_opts else 0
            curr = st.selectbox(
                "Preferred Currency",
                _currency_opts,
                index=_currency_idx
            )
            
            if st.form_submit_button("Update Business Profile"):
                upsert_business_profile(user_id, b_name, o_name, curr)
                st.session_state['business_name'] = b_name
                st.session_state['owner_name'] = o_name
                st.session_state['currency'] = curr
                st.session_state['currency_symbol'] = get_currency_symbol(curr)
                st.success("Business profile updated successfully!")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("### Session Management")
        st.write(f"Signed in as: **{username}**")
        if st.button("🚪 Sign Out of DemandWise"):
            st.session_state['authenticated'] = False
            st.session_state['user_id'] = None
            st.session_state['username'] = None
            st.session_state['currency_symbol'] = "$"
            st.success("Signed out successfully.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. Main Dispatcher
# -----------------------------------------------------------------------------
def main():
    if not st.session_state.get('authenticated'):
        render_auth_view()
    else:
        render_main_app()


if __name__ == '__main__':
    main()
