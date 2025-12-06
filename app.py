import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from dotenv import load_dotenv
from datetime import datetime

# Load env
load_dotenv()

import analysis as an
import visualization as vz

# Optional modules
try:
    import cleaner
    CLEANER_AVAILABLE = True
except Exception:
    CLEANER_AVAILABLE = False

try:
    import llmutil as ai
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

try:
    import login_systemusingmysql as auth
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False


# =======================================================================================
# 🌟 NEW HELPER FUNCTION — Show two charts in one row
# =======================================================================================
def show_two_charts(fig1, fig2):
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)


# ---------- Streamlit page config ----------
st.set_page_config(page_title="ProfitLens - Integrated App", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0078d4 0%, #106ebe 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border: 1px solid #dee2e6; border-radius: 10px; padding: 1.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #0078d4; margin-bottom: 0.5rem; }
    .metric-label { font-size: 0.9rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    .sidebar-section { background: #f8f9fa; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border-left: 4px solid #0078d4; }
    .mapping-container { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid #2196f3; }
    .control-panel { background: #f0f2f6; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #e0e2e6; }
</style>
""", unsafe_allow_html=True)


# ---------- Session state init ----------
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = not AUTH_AVAILABLE
    if 'username' not in st.session_state:
        st.session_state.username = "Guest" if not AUTH_AVAILABLE else ""
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'mapping_complete' not in st.session_state:
        st.session_state.mapping_complete = False
    if 'view_type' not in st.session_state:
        st.session_state.view_type = 'visual'
    if 'top_n_products' not in st.session_state:
        st.session_state.top_n_products = 5
    if 'top_n_customers' not in st.session_state:
        st.session_state.top_n_customers = 5
    if 'top_n_categories' not in st.session_state:
        st.session_state.top_n_categories = 5

init_session_state()


# UTIL FUNCTIONS (unchanged)
def safe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()


def auto_detect_mapping(df: pd.DataFrame):
    mapping = {}
    cols = df.columns.tolist()
    lower = [c.lower().replace("_"," ").replace("-", " ").strip() for c in cols]
    for i, c in enumerate(lower):
        raw = cols[i]
        if 'txn' in c or 'transaction' in c or 'order' in c:
            mapping[raw] = 'Transaction ID'
        elif c in ('date', 'purchase date', 'order date', 'datetime', 'time'):
            mapping[raw] = 'Date'
        elif 'cust' in c or 'customer' in c:
            mapping[raw] = 'Customer ID'
        elif c in ('gender','sex'):
            mapping[raw] = 'Gender'
        elif c == 'age':
            mapping[raw] = 'Age'
        elif 'product' in c and ('category' in c or 'type' in c):
            mapping[raw] = 'Product Category'
        elif c in ('qty','quantity','count'):
            mapping[raw] = 'Quantity'
        elif 'price' in c:
            mapping[raw] = 'Price per Unit'
        elif 'total' in c or 'amount' in c:
            mapping[raw] = 'Total Amount'
    return mapping


# COLUMN MAPPING UI (unchanged)
def user_column_mapping_interface(df):
    st.markdown('<div class="mapping-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Column Mapping")

    standard_columns = [
        "Transaction ID","Date","Customer ID","Gender","Age",
        "Product Category","Quantity","Price per Unit","Total Amount"
    ]
    auto_map = auto_detect_mapping(df)

    st.write(auto_map if auto_map else "No automatic mappings detected.")
    mapping = {}

    for std in standard_columns:
        options = ["-- Not selected --"] + df.columns.tolist()
        default = 0
        for raw, mapped in auto_map.items():
            if mapped == std:
                default = options.index(raw)
                break

        sel = st.selectbox(f"Map '{std}'", options, index=default, key=f"map_{std}")
        if sel != "-- Not selected --":
            mapping[sel] = std

    st.markdown('</div>', unsafe_allow_html=True)
    return mapping


# DATA LOADING (unchanged)
def load_and_process_data_sidebar():
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose CSV or Excel file", type=['csv','xlsx','xls'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.raw_df = df
            st.success(f"Loaded {uploaded_file.name} — shape {df.shape}")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")


# APPLY CLEANING (unchanged)
def apply_user_column_mapping(df, mapping):
    mapped = df.rename(columns=mapping)
    if CLEANER_AVAILABLE:
        return cleaner.clean_dataframe_simple(mapped)
    return mapped


# CONTROL PANEL (unchanged)
def show_control_panel():
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])

    with col1:
        st.session_state.top_n_products = st.slider("Top N Products", 1, 50, st.session_state.top_n_products)

    with col2:
        vt = st.radio("Display as", ["📊 Visual", "📋 Table"], horizontal=True)
        st.session_state.view_type = 'visual' if '📊' in vt else 'table'
    
    st.markdown('</div>', unsafe_allow_html=True)


# KPI SECTION (unchanged)
def show_kpis(df):
    summary = an.total_sales_summary(df)
    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric("Total Sales", f"${summary['Total sales']:,.0f}")
    with col2:
        st.metric("Units Sold", f"{summary['Total quantities sold']:,}")
    with col3:
        st.metric("Transactions", f"{summary['Total Transaction']:,}")
    with col4:
        st.metric("Avg Basket Size", f"{an.average_basket_size(df)}")


# =======================================================================================
# ⭐ UPDATED PRODUCT ANALYSIS PANEL — SIDE-BY-SIDE CHARTS ADDED
# =======================================================================================
def product_analysis_panel(df):
    st.header("🛍️ Product Analysis")

    cats = sorted(df['Product Category'].dropna().unique())
    selected_cats = st.multiselect("Filter categories", options=cats, default=cats[:10])
    filtered = df[df['Product Category'].isin(selected_cats)]

    topn = st.session_state.top_n_products

    st.subheader(f"Top {topn} Products – Revenue & Quantity")

    if st.session_state.view_type == 'visual':
        
        fig_qty = vz.visualize_top_selling_products(filtered, n=topn)
        fig_rev = vz.visualize_top_selling_products_by_amount(filtered, n=topn)

        # ⭐ NEW — Show side by side
        show_two_charts(fig_qty, fig_rev)

    else:
        st.dataframe(an.top_selling_products(filtered, n=topn).reset_index())
        st.dataframe(an.product_amount_wise_sales(filtered, n=topn).reset_index())


# CUSTOMER ANALYSIS (unchanged)
def customer_analysis_panel(df):
    st.header("👥 Customer Analysis")
    # ... unchanged ...


# TRENDS PANEL (unchanged)
def trends_panel(df):
    st.header("📅 Trends & Forecasting")
    # ... unchanged ...


# CATEGORY PANEL (unchanged)
def category_panel(df):
    st.header("📂 Category Breakdown")
    # ... unchanged ...


# PROFIT PANEL (unchanged)
def profit_inventory_panel(df):
    st.header("💰 Profit & Inventory")
    # ... unchanged ...


# EXPORT PANEL (unchanged)
def export_panel(df):
    st.header("📤 Export & Reports")
    # ... unchanged ...


# LOGIN & MAIN APP (unchanged)
def show_login_page():
    pass  # unchanged


def app_main():
    load_and_process_data_sidebar()

    if st.session_state.raw_df is None:
        st.info("Upload data to continue.")
        return

    if not st.session_state.mapping_complete:
        st.write("### Column Mapping")
        mapping = user_column_mapping_interface(st.session_state.raw_df)
        if st.button("Apply Mapping"):
            cleaned = apply_user_column_mapping(st.session_state.raw_df, mapping)
            st.session_state.cleaned_df = cleaned
            st.session_state.mapping_complete = True
            st.rerun()
        return

    df = st.session_state.cleaned_df.copy()

    show_control_panel()
    show_kpis(df)

    tabs = st.tabs(["Products","Customers","Trends","Categories","Profit & Inventory","Export"])

    with tabs[0]:
        product_analysis_panel(df)
    with tabs[1]:
        customer_analysis_panel(df)
    with tabs[2]:
        trends_panel(df)
    with tabs[3]:
        category_panel(df)
    with tabs[4]:
        profit_inventory_panel(df)
    with tabs[5]:
        export_panel(df)


def main():
    app_main()


if __name__ == "__main__":
    main()
