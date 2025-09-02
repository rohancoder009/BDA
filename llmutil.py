# llmutil.py
import os
import re
import json
import numpy as np
import pandas as pd

try:
    import google.generativeai as genai
except Exception:
    genai = None

from dotenv import load_dotenv
load_dotenv()


# ---------------- Setup ----------------
def setup_gemini(api_key=None):
    """Configure Gemini (optional). If genai is not installed this will silently skip."""
    if genai is None:
        print("⚠️ google.generativeai not available. LLM calls will be skipped.")
        return False
    if api_key is None:
        api_key = os.getenv("API_KEY")
    genai.configure(api_key=api_key)
    return True


# ---------------- Safe serializer ----------------
def safe_json_serialize(obj):
    """Convert pandas/numpy/python objects into JSON-safe native Python types."""
    if isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json_serialize(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(obj, (pd.Series,)):
        return obj.tolist()
    if obj is None:
        return None
    try:
        # for datetime-like objects with isoformat
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
    except Exception:
        pass
    # fallback to str
    return str(obj)


# ---------------- LLM-based mapping (optional) ----------------
def _clean_samples_for_prompt(series):
    """Return up to 5 short stringified sample values for prompt (safe types)."""
    vals = series.dropna().astype(str).head(5).unique()
    vals = list(vals)[:5]
    cleaned = []
    for v in vals:
        v = v.strip()
        if len(v) > 80:
            v = v[:77] + "..."
        cleaned.append(v)
    return cleaned


def get_column_mapping(df: pd.DataFrame, standard_cols: list, use_llm=True):
    """
    Ask Gemini to map original columns to standard columns.
    Returns dict mapping original_column -> standard_column OR {} on failure.
    """
    # Quick exit if genai not available or user disabled LLM
    if not use_llm or genai is None:
        return {}

    column_info = []
    for col in df.columns:
        samples = _clean_samples_for_prompt(df[col])
        column_info.append({
            "column_name": str(col),
            "sample_values": samples,
            "data_type": str(df[col].dtype),
            "non_null_count": int(df[col].count())
        })

    prompt = f"""
You are a careful data engineer. Map the columns described below to the provided STANDARD column names.
Return a single JSON object mapping exact original column names to one of the STANDARD columns or the value "None".
Do NOT include any explanatory text.

COLUMNS:
{json.dumps(column_info, indent=2)}

STANDARD_COLUMNS:
{json.dumps(standard_cols, indent=2)}

RULES:
Most important dont repeat columns and if Any column name is given exactly no need to change it you are changing customer ID many times please ensure proper rules
1) Map only when you're at least 90% confident; otherwise return "None" for that column.
2) Each original column must map to exactly one value (a standard column name or "None").
3) Respond ONLY with a JSON object, for example:
4) Dont duplicates column name give a column name for only  needed data  
5) dont change Invoice No. column name 
{{ "orig_col_name": "Total Amount", "other_col": "None" }}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Remove triple backticks/code fences if any
        text = re.sub(r'```(?:json)?', '', text)
        text = text.strip('` \n\t')
        # Extract first JSON object substring
        m = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if not m:
            print("❌ LLM returned non-JSON. Skipping LLM mapping.")
            return {}
        json_text = m.group(0)
        mapping = json.loads(json_text)
        # Validate mapping keys and values
        clean_map = {}
        for orig, std in mapping.items():
            if orig in df.columns and (std in standard_cols or std == "None"):
                if std != "None":
                    clean_map[orig] = std
        return clean_map
    except Exception as e:
        print(f"❌ Gemini/LLM error: {e}")
        return {}


# ---------------- Rule-based fallback (exposed) ----------------
def fallback_rule_based_mapping(df_columns, standard_cols):
    """Simple but robust keyword-based mapping. Accepts list-like df_columns."""
    # patterns tuned to analysis.py standard columns
    patterns = {
        "Total Amount": ["total_amount", "totalamount", "amount", "amt", "sales", "revenue", "total", "order_total", "grand_total"],
        "Quantity": ["qty", "quantity", "units", "count", "items", "pieces", "sold", "no_of_items"],
        "Product Category": ["category", "product_category", "product", "item", "prod_cat", "product_type"],
        "Date": ["date", "order_date", "txn", "transaction_date", "created_at", "timestamp", "datetime"],
        "Customer ID": ["customer_id", "cust_id", "customerid", "user_id", "account_id", "client_id", "buyer_id"],
        "Gender": ["gender", "sex", "m_f"],
        "Age": ["age", "years", "age_year", "buyer_age"],
        "Price per Unit": ["unit_price", "price_per_unit", "price", "unit_cost", "rate", "price_each"],
        "Transaction ID": ["transaction_id", "order_id", "receipt_id", "invoice_id", "id", "txn_id"]
    }

    used = set()
    mapping = {}
    for orig in df_columns:
        norm = str(orig).strip().lower().replace(" ", "_").replace("-", "_")
        best = None
        for std, keywords in patterns.items():
            if std in used:
                continue
            for kw in keywords:
                if norm == kw or kw in norm or norm in kw:
                    best = std
                    break
            if best:
                break
        if best and best in standard_cols and best not in used:
            mapping[orig] = best
            used.add(best)
    return mapping


# ---------------- Hybrid util ----------------
def get_final_column_mapping(df: pd.DataFrame, standard_cols: list, prefer_llm=True):
    """
    Try LLM mapping first (if available), then fallback_rule_based. Returns a stable mapping.
    """
    # 1) try LLM
    mapping = {}
    if prefer_llm and genai is not None:
        mapping = get_column_mapping(df, standard_cols, use_llm=True)
        if mapping:
            # ensure no duplicate target names (dedupe)
            inv = {}
            for orig, std in mapping.items():
                if std in inv:
                    # conflict - keep earlier assigned, skip this one
                    continue
                inv[std] = orig
            mapping = {orig: std for orig, std in mapping.items() if std in inv}
    # 2) fallback
    if not mapping:
        mapping = fallback_rule_based_mapping(df.columns, standard_cols)

    return mapping
def generate_summary(analysis_data, df_info):
    """Generate business summary using Gemini"""
    
    # Convert analysis data to JSON-safe format
    safe_data = safe_json_serialize(analysis_data)
    
    prompt = f"""
Create a concise business report from this data analysis:

DATASET INFO:
- Rows: {df_info.get('rows', 'N/A')}
- Date Range: {df_info.get('date_range', 'N/A')}

ANALYSIS RESULTS:
{json.dumps(safe_data, indent=2)}

Create a professional business summary with:
1. Executive Summary (2-3 sentences)
2. Key Insights (3-4 bullet points)  
3. Recommendations (2-3 actionable items)

Keep it concise and business-focused.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating summary: {e}"