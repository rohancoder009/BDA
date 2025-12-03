# app2.py - Final integrated version using analysis.py and visualization.py
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

# Import analysis & visualization (these are the files you provided)
import analysis as an   # functions for analysis (total_sales_summary, product_amount_wise_sales, ...)
import visualization as vz  # helper plotting functions returning matplotlib figs

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
# for MySql

try:
    import login_systemusingmysql as auth
    AUTH_AVAILABLE = True
except Exception:
    AUTH_AVAILABLE = False 
# For SQL lite
#try:
  #  import login_system as auth
   # auth.init_database()  # creates SQLite DB automatically
   # AUTH_AVAILABLE = True
#except Exception:
 #   AUTH_AVAILABLE = False

# ---------- Streamlit page config ----------
st.set_page_config(page_title="ProfitLens - Integrated App", page_icon="📊", layout="wide")

# ---------- CSS (kept as in your file) ----------
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
        st.session_state.view_type = 'visual'  # 'visual' or 'table'
    if 'top_n_products' not in st.session_state:
        st.session_state.top_n_products = 5
    if 'top_n_customers' not in st.session_state:
        st.session_state.top_n_customers = 5
    if 'top_n_categories' not in st.session_state:
        st.session_state.top_n_categories = 5
    if 'saved_reports' not in st.session_state:
        st.session_state.saved_reports = {}

init_session_state()

# ---------- Utilities ----------
def safe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()

# Try to auto-detect typical column names and propose mapping
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
        elif 'cust' in c or 'customer' in c or 'client' in c:
            mapping[raw] = 'Customer ID'
        elif c in ('gender','sex'):
            mapping[raw] = 'Gender'
        elif c == 'age':
            mapping[raw] = 'Age'
        elif 'product' in c and ('category' in c or 'type' in c):
            mapping[raw] = 'Product Category'
        elif c in ('qty','quantity','count'):
            mapping[raw] = 'Quantity'
        elif 'price' in c and ('unit' in c or 'per' in c):
            mapping[raw] = 'Price per Unit'
        elif 'total' in c or 'amount' in c or 'revenue' in c:
            mapping[raw] = 'Total Amount'
    return mapping

# ---------- Column mapping UI ----------
def user_column_mapping_interface(df):
    st.markdown('<div class="mapping-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Column Mapping")

    standard_columns = [
        "Transaction ID","Date","Customer ID","Gender","Age",
        "Product Category","Quantity","Price per Unit","Total Amount"
    ]
    auto_map = auto_detect_mapping(df)
    st.markdown("**Auto-detected mappings (editable)**")
    st.write(auto_map if auto_map else "No automatic mappings detected.")
    mapping = {}

    for std in standard_columns:
        options = ["-- Not selected --"] + df.columns.tolist()
        default_index = 0
        # if auto detected
        for raw, mapped in auto_map.items():
            if mapped == std:
                default_index = options.index(raw) if raw in options else 0
                break
        sel = st.selectbox(f"Map '{std}'", options, index=default_index, key=f"map_{std}")
        if sel and sel != "-- Not selected --":
            mapping[sel] = std

    st.markdown('</div>', unsafe_allow_html=True)
    return mapping

# ---------- Data load and sample ----------
def load_and_process_data_sidebar():
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose CSV or Excel file", type=['csv','xlsx','xls'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.raw_df = df
            st.success(f"Loaded {uploaded_file.name} — shape {df.shape}")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    if st.sidebar.button("Load Sample Data"):
        st.session_state.raw_df = create_sample_data(500)
        st.success("Sample data loaded (500 rows)")

def create_sample_data(n_records=500):
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_records, freq='D')
    categories = ['Electronics','Clothing','Home & Garden','Sports','Books']
    genders = ['M','F']
    data = {
        'Transaction ID': [f'TXN{i:06d}' for i in range(1, n_records+1)],
        'Date': np.random.choice(dates, n_records),
        'Customer ID': [f'CUST{np.random.randint(1000,9999)}' for _ in range(n_records)],
        'Gender': np.random.choice(genders, n_records),
        'Age': np.random.randint(18,75,n_records),
        'Product Category': np.random.choice(categories, n_records),
        'Quantity': np.random.randint(1,6,n_records),
        'Price per Unit': np.round(np.random.uniform(10,500,n_records),2),
    }
    df = pd.DataFrame(data)
    df['Total Amount'] = (df['Quantity'] * df['Price per Unit']).round(2)
    return df

# ---------- Apply mapping and cleaning ----------
def apply_user_column_mapping(df, mapping):
    if not mapping or len(mapping)==0:
        st.error("No mapping provided.")
        return None
    mapped = df.rename(columns=mapping)
    # try cleaner functions if available
    if CLEANER_AVAILABLE and hasattr(cleaner, 'clean_dataframe_simple'):
        try:
            cleaned = cleaner.clean_dataframe_simple(mapped, prefer_llm=False)
            return cleaned
        except Exception:
            # fallback to mapped
            return mapped
    else:
        # ensure standard columns exist (fill missing with None)
        for c in ["Transaction ID","Date","Customer ID","Gender","Age","Product Category","Quantity","Price per Unit","Total Amount"]:
            if c not in mapped.columns:
                mapped[c] = None
        # convert Date to datetime if possible
        try:
            mapped['Date'] = pd.to_datetime(mapped['Date'])
        except Exception:
            pass
        return mapped

# ---------- Control panel ----------
def show_control_panel():
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("### 🎚️ Controls")
        st.session_state.top_n_products = st.slider("Top N Products", 1, 50, st.session_state.top_n_products)
        st.session_state.top_n_customers = st.slider("Top N Customers", 1, 50, st.session_state.top_n_customers)
        st.session_state.top_n_categories = st.slider("Top N Categories", 1, 20, st.session_state.top_n_categories)
    with col2:
        st.markdown("### 👁 View")
        vt = st.radio("Display as", ["📊 Visual", "📋 Table"], horizontal=True)
        st.session_state.view_type = 'visual' if '📊' in vt else 'table'
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Analysis display ----------
def show_kpis(df):
    try:
        summary = an.total_sales_summary(df)
    except Exception:
        summary = {'Total sales':0,'Total quantities sold':0,'Total Transaction':0}
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>${summary['Total sales']:,.0f}</div><div class='metric-label'>Total Sales</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{summary['Total quantities sold']:,}</div><div class='metric-label'>Units Sold</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{summary['Total Transaction']:,}</div><div class='metric-label'>Transactions</div></div>", unsafe_allow_html=True)
    with c4:
        try:
            avg_basket = an.average_basket_size(df)
        except Exception:
            avg_basket = 0
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_basket}</div><div class='metric-label'>Avg Basket Size</div></div>", unsafe_allow_html=True)

# ---------- Product analysis (uses functions from analysis.py & visualization.py) ----------
def product_analysis_panel(df):
    st.header("🛍️ Product Analysis")

    # category filter
    cats = sorted(df['Product Category'].dropna().unique().tolist())
    selected_cats = st.multiselect("Filter categories", options=cats, default=cats[:min(len(cats),10)])
    filtered = df[df['Product Category'].isin(selected_cats)]

    # Choose metric
    metric = st.selectbox("Metric", ["Revenue","Quantity","Both"])
    # Top N
    topn = st.session_state.top_n_products

    if metric in ("Revenue","Both"):
        st.subheader(f"Top {topn} Products by Revenue")
        # use analysis function product_amount_wise_sales
        rev = an.product_amount_wise_sales(filtered, n=topn)
        # rev is a Series indexed by Product Category
        if st.session_state.view_type == 'visual':
            # use visualization helper
            try:
                fig = vz.visualize_top_selling_products_by_amount(filtered, n=topn)
                st.pyplot(fig)
            except Exception:
                # fallback simple matplotlib
                rev_df = rev.reset_index()
                fig, ax = plt.subplots(figsize=(10,6))
                ax.barh(rev_df['Product Category'], rev_df['Total Amount'] if 'Total Amount' in rev_df.columns else rev_df[rev.name])
                ax.set_xlabel("Revenue")
                st.pyplot(fig)
        else:
            rev_df = rev.reset_index()
            rev_df.columns = ['Product Category','Total Amount'] if len(rev_df.columns)>=2 else ['Product Category','Total Amount']
            st.dataframe(rev_df)

    if metric in ("Quantity","Both"):
        st.subheader(f"Top {topn} Products by Quantity")
        qty = an.top_selling_products(filtered, n=topn)
        if st.session_state.view_type == 'visual':
            try:
                fig = vz.visualize_top_selling_products(filtered, n=topn)
                st.pyplot(fig)
            except Exception:
                qty_df = qty.reset_index()
                fig, ax = plt.subplots(figsize=(10,6))
                ax.barh(qty_df['Product Category'], qty_df['Quantity'] if 'Quantity' in qty_df.columns else qty_df[qty.name])
                ax.set_xlabel("Units Sold")
                st.pyplot(fig)
        else:
            qty_df = qty.reset_index()
            qty_df.columns = ['Product Category','Quantity']
            st.dataframe(qty_df)

    # Product monthly performance (heatmap/line)
    if st.checkbox("Show product monthly performance"):
        try:
            perf = an.product_monthly_performance(filtered)
            if st.session_state.view_type == 'visual':
                fig = vz.visualize_product_monthly_performance(filtered)
                st.pyplot(fig)
            else:
                st.dataframe(perf.head(200))
        except Exception as e:
            st.error(f"Error product monthly perf: {e}")

    # Price elasticity
    if st.checkbox("Show price elasticity by category"):
        try:
            elast = an.price_elasticity(filtered)
            if st.session_state.view_type == 'visual':
                fig = vz.visualize_price_elasticity(filtered)
                st.pyplot(fig)
            else:
                st.dataframe(elast)
        except Exception as e:
            st.error(f"Error elasticity: {e}")

# ---------- Customer analysis ----------
def customer_analysis_panel(df):
    st.header("👥 Customer Analysis")

    topn = st.session_state.top_n_customers
    # Top customers
    st.subheader(f"Top {topn} Customers by Revenue")
    top_cust = an.top_customers_by_sales(df, n=topn)
    # top_cust is Series indexed by Customer ID
    if st.session_state.view_type == 'visual':
        try:
            fig = vz.visualize_top_customers(df, n=topn)
            st.pyplot(fig)
        except Exception:
            df_tc = top_cust.reset_index()
            df_tc.columns = ['Customer ID','Total Amount']
            fig, ax = plt.subplots(figsize=(10,6))
            ax.barh(df_tc['Customer ID'], df_tc['Total Amount'])
            st.pyplot(fig)
    else:
        st.dataframe(top_cust.reset_index().rename(columns={top_cust.name: 'Total Amount'}))

    # New vs returning
    st.subheader("New vs Returning Customers")
    newret = an.new_vs_returning_customers(df)
    if st.session_state.view_type == 'visual':
        try:
            fig = vz.visualize_new_vs_returning(df)
            st.pyplot(fig)
        except Exception:
            fig, ax = plt.subplots()
            ax.bar(newret.index.astype(str), newret.values)
            st.pyplot(fig)
    else:
        st.dataframe(newret.reset_index().rename(columns={0:'Count'}))

    # RFM segment (table)
    if st.checkbox("Show RFM segmentation (table)"):
        try:
            rfm = an.rfm_segmentation(df)
            st.dataframe(rfm.head(200))
        except Exception as e:
            st.error(f"RFM error: {e}")

    # CLV & retention
    try:
        clv = an.customer_lifetime_value(df)
        st.metric("Estimated CLV (approx)", f"{clv:,.2f}")
    except Exception:
        pass
    try:
        retention = an.customer_retention_rate(df)
        if st.session_state.view_type == 'visual':
            try:
                fig = vz.visualize_retention_rate(df)
                st.pyplot(fig)
            except Exception:
                st.line_chart(retention)
        else:
            st.dataframe(retention.reset_index().rename(columns={retention.name:'Retention %'}))
    except Exception:
        pass

# ---------- Trends panel ----------
def trends_panel(df):
    st.header("📅 Trends & Forecasting")
    # monthly trend
    st.subheader("Monthly Sales Trend")
    monthly = an.monthly_sales_trend(df)
    if st.session_state.view_type == 'visual':
        try:
            fig = vz.visualize_monthly_sales_trend(df)
            st.pyplot(fig)
        except Exception:
            st.line_chart(monthly.set_index('Month')['Total Amount'])
    else:
        st.dataframe(monthly)

    # Growth
    st.subheader("Month-over-month Growth")
    growth = an.monthly_growth_rate(df)
    if st.session_state.view_type == 'visual':
        try:
            fig = vz.visualize_monthly_growth_rate(df)
            st.pyplot(fig)
        except Exception:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.bar(growth['Month'], growth['Growth %'])
            st.pyplot(fig)
    else:
        st.dataframe(growth)

    # Moving average & forecast
    window = st.slider("Moving average window (days)", 3, 30, 7)
    ma = an.moving_average_sales(df, window=window)
    st.subheader(f"{window}-day moving average (daily)")
    if st.session_state.view_type == 'visual':
        try:
            fig = vz.visualize_moving_average(df, window=window)
            st.pyplot(fig)
        except Exception:
            st.line_chart(ma)
    else:
        st.dataframe(ma.reset_index())

    # Forecast
    periods = st.number_input("Forecast months", min_value=1, max_value=36, value=6)
    forecast = an.sales_forecast(df, periods=periods)
    st.subheader("Simple forecast (projection)")
    if st.session_state.view_type == 'visual':
        try:
            fig = vz.visualize_sales_forecast(df, periods=periods)
            st.pyplot(fig)
        except Exception:
            st.line_chart(forecast.values)
    else:
        st.dataframe(pd.DataFrame({'Period': range(1,len(forecast)+1), 'Forecast': forecast.values}))

    # Anomaly detection
    if st.checkbox("Show anomalies (daily)"):
        try:
            anomalies = an.anomaly_detection(df)
            if st.session_state.view_type == 'visual':
                fig = vz.visualize_sales_anomalies(df)
                st.pyplot(fig)
            else:
                st.dataframe(anomalies.reset_index().rename(columns={0:'Sales'}))
        except Exception as e:
            st.error(f"Anomaly detection error: {e}")

# ---------- Category breakdown ----------
def category_panel(df):
    st.header("📂 Category Breakdown")
    cat_summary = df.groupby('Product Category')['Total Amount'].sum().reset_index().sort_values('Total Amount', ascending=False)
    topk = st.session_state.top_n_categories
    topcats = cat_summary.head(topk)
    st.subheader(f"Top {topk} Categories")
    if st.session_state.view_type == 'visual':
        try:
            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie(topcats['Total Amount'], labels=topcats['Product Category'], autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Top {topk} Categories by Sales")
            st.pyplot(fig)
        except Exception:
            st.dataframe(topcats)
    else:
        st.dataframe(topcats)

# ---------- Profit & inventory ----------
def profit_inventory_panel(df):
    st.header("💰 Profit & Inventory")
    if 'Cost Price' in df.columns or 'Cost' in df.columns:
        try:
            profit = an.total_profit(df)
            margin = an.profit_margin(df)
            st.metric("Total Profit", f"{profit:,.2f}")
            st.metric("Profit Margin %", f"{margin:.2f}%")
            try:
                fig = vz.visualize_total_profit(df)
                st.pyplot(fig)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Profit calc error: {e}")
    else:
        st.info("No Cost Price column available. Add 'Cost Price' column to analyze profits.")

    inv_file = st.file_uploader("Upload inventory (CSV with Product Category, Stock)", type=['csv'])
    if inv_file:
        try:
            inv_df = pd.read_csv(inv_file)
            risk = an.stock_out_risk(df, inv_df)
            st.dataframe(risk)
            try:
                fig = vz.visualize_stock_out_risk(df, inv_df)
                st.pyplot(fig)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Inventory read error: {e}")

# ---------- Export & Reports ----------
def export_panel(df):
    st.header("📤 Export & Reports")

    # Download cleaned data
    csv_bytes = safe_to_csv_bytes(df)
    st.download_button(
        "Download filtered dataset (CSV)",
        data=csv_bytes,
        file_name="profitlens_filtered.csv",
        mime="text/csv"
    )

    st.markdown("### 🤖 AI-Generated Business Report (Gemini)")

    if LLM_AVAILABLE and st.button("Generate AI Business Report"):
        try:
            ai_report = generate_ai_summary_report(df)

            st.markdown("### 📘 AI Business Report")
            st.markdown(ai_report)

            st.download_button(
                "Download AI Report (Markdown)",
                data=ai_report,
                file_name="ProfitLens_AI_Business_Report.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"AI report generation failed: {e}")

    elif not LLM_AVAILABLE:
        st.warning("LLM module not available. Cannot generate AI-powered report.")


def generate_ai_summary_report(df):
    """
    Use Gemini to generate a business intelligence report based on the dataset.
    """

    # Prepare analysis data for Gemini
    try:
        analysis_results = {
            "kpis": an.total_sales_summary(df),
            "top_products_qty": an.top_selling_products(df, 5).to_dict(),
            "top_products_revenue": an.product_amount_wise_sales(df, 5).to_dict(),
            "monthly_sales": an.monthly_sales_trend(df).set_index("Month")["Total Amount"].to_dict(),
            "category_breakdown": df.groupby("Product Category")["Total Amount"].sum().to_dict()
        }
    except Exception as e:
        raise Exception(f"Analysis extraction failed: {e}")

    # Metadata
    df_info = {
        "rows": len(df),
        "date_range": f"{df['Date'].min().date()} → {df['Date'].max().date()}"
    }

    # Call Gemini through llmutil
    try:
        ai_report = ai.generate_summary(analysis_results, df_info)
        return ai_report

    except Exception as e:
        raise Exception(f"Gemini summary generation failed: {e}")

# ---------- Main app ----------
def show_login_page():
    st.markdown('<div class="main-header">🔐 ProfitLens Login</div>', unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(["Login","Sign up"])
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                btn = st.form_submit_button("Login")
                if btn and AUTH_AVAILABLE:
                    user = auth.check_user(username,password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Logged in")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        with tab2:
            with st.form("signup_form"):
                un = st.text_input("Choose username")
                pw = st.text_input("Choose password", type="password")
                pw2 = st.text_input("Confirm password", type="password")
                btn2 = st.form_submit_button("Create account")
                if btn2 and AUTH_AVAILABLE:
                    if pw != pw2:
                        st.error("Passwords don't match")
                    else:
                        ok = auth.add_user(un, pw)
                        if ok:
                            st.success("Account created")
                        else:
                            st.error("Username exists")

def app_main():
    st.markdown('<div class="main-header">📊 ProfitLens - Business Data Analyzer</div>', unsafe_allow_html=True)

    # Sidebar: data upload & global filters
    with st.sidebar:
        st.markdown("### Data Controls")
        load_and_process_data_sidebar()
        st.markdown("---")
        if st.session_state.raw_df is not None:
            st.markdown("### Quick actions")
            if st.button("Clear data"):
                for k in ['raw_df','cleaned_df','column_mapping','mapping_complete']:
                    if k in st.session_state:
                        del st.session_state[k]
                    st.rerun()
    if st.session_state.raw_df is None:
        st.info("Upload CSV/XLSX or load sample data to begin.")
        return

    # Mapping step
    if not st.session_state.mapping_complete:
        st.markdown("## 🔧 Step 1: Column mapping")
        st.write("Preview of uploaded data (first 5 rows):")
        st.dataframe(st.session_state.raw_df.head())
        mapping = user_column_mapping_interface(st.session_state.raw_df)
        if st.button("Apply mapping and clean data"):
            mapped = apply_user_column_mapping(st.session_state.raw_df, mapping)
            if mapped is not None:
                st.session_state.column_mapping = mapping
                st.session_state.cleaned_df = mapped
                st.session_state.mapping_complete = True
                st.success("Mapping applied and data cleaned.")
                st.rerun()
        return

    # Now have cleaned_df
    df = st.session_state.cleaned_df.copy()
    # Ensure Date column name is 'Date' and is datetime
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            pass

    # Global filters
    st.sidebar.markdown("### Global Filters")
    try:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        dr = st.sidebar.date_input("Date range", [min_date, max_date])
        if isinstance(dr, list) and len(dr)==2:
            start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
            df = df[(df['Date']>=start) & (df['Date']<=end)]
    except Exception:
        pass

    # Show controls, KPIs, tabs
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

# Run
def main():
    init_session_state()
    if not st.session_state.logged_in and AUTH_AVAILABLE:
        show_login_page()
    else:
        app_main()

if __name__ == "__main__":
    main()
