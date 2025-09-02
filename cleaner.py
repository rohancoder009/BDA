# cleaner.py
import pandas as pd
import numpy as np
import re
from typing import Dict, List
import llmutil as ai

# Standard columns expected by analysis.py
STANDARD_COLS = [
    "Transaction ID", "Date", "Customer ID", "Gender", "Age",
    "Product Category", "Quantity", "Price per Unit", "Total Amount"
]


# ---------------- Normalization helpers ----------------
def normalize_name(name: str) -> str:
    """Make a normalized underscore_separated column name."""
    name = str(name).strip().lower()
    name = re.sub(r'[^0-9a-z]+', '_', name)
    name = re.sub(r'__+', '_', name)
    return name.strip('_')


def infer_by_samples(series: pd.Series) -> str:
    """Heuristic to label a column from sample values."""
    s = series.dropna().astype(str).head(10).astype(str)
    if s.empty:
        return ""
    # detect date-like (many parsable dates)
    parsed = 0
    for v in s:
        v = v.strip()
        # common date patterns
        if re.search(r'\d{4}-\d{2}-\d{2}', v) or re.search(r'\d{2}/\d{2}/\d{4}', v) or re.search(r'\d{1,2}\s[a-zA-Z]{3,}\s\d{2,4}', v):
            parsed += 1
        # also try numeric timestamp-ish
    if parsed >= max(1, len(s)//2):
        return "Date"

    # detect numeric currency amounts (dollar/rupee sign or decimals)
    numeric_like = 0
    currency_sign = 0
    for v in s:
        v2 = v.replace(',', '').replace('₹', '').replace('$', '').replace('rs', '').strip()
        if re.match(r'^\-?\d+(\.\d+)?$', v2):
            numeric_like += 1
        if re.search(r'[\$\₹]|rs\.?', v.lower()):
            currency_sign += 1
    if numeric_like >= max(1, len(s)//2) and (currency_sign > 0 or any('.' in v for v in s)):
        return "Total Amount"

    # detect small integers (likely quantity)
    ints = 0
    for v in s:
        v2 = v.replace(',', '')
        if re.match(r'^\d+$', v2) and int(v2) < 10000:
            ints += 1
    if ints >= max(1, len(s)//2):
        return "Quantity"

    # detect short text categories
    unique_vals = len(set(s.tolist()))
    avg_len = sum(len(x) for x in s) / len(s)
    if unique_vals <= len(s) and avg_len <= 30:
        # candidate for product category / gender / id
        # gender detection
        low = [x.lower() for x in s]
        if any(x in ('male','female','m','f') for x in low):
            return "Gender"
        # customer id heuristic: long alphanumeric or prefix 'CUST'
        if any(re.search(r'(cust|user|client|acct)', x.lower()) for x in s) or any(re.match(r'^[A-Z0-9\-_]{6,}$', x) for x in s):
            return "Customer ID"
        # else product category
        return "Product Category"

    return ""


# ---------------- Fallback pattern mapping (exposed) ----------------
def smart_fallback_mapping(df_columns) -> Dict[str, str]:
    """
    Robust pattern-based mapping. Returns original_name -> Standard Name mapping.
    """
    return ai.fallback_rule_based_mapping(df_columns, STANDARD_COLS)


# ---------------- Main mapping pipeline ----------------
def map_columns_hybrid(df: pd.DataFrame, prefer_llm=True) -> Dict[str, str]:
    """
    1) Try hybrid LLM + fallback using ai.get_final_column_mapping
    2) If still missing, run sample-value heuristics
    Returns a mapping dict original -> standard
    """
    # 1) LLM/fallback rule-based from llmutil
    mapping = ai.get_final_column_mapping(df, STANDARD_COLS, prefer_llm)

    # 2) Ensure keys are existing original columns and avoid duplicates
    mapping = {k: v for k, v in mapping.items() if k in df.columns}

    used_targets = set(mapping.values())

    # 3) For columns not mapped yet, try heuristics on samples + normalized name
    for col in df.columns:
        if col in mapping:
            continue
        # check normalized name first
        norm = normalize_name(col)
        # try direct pattern match among STANDARD_COLS
        matched = None
        for std in STANDARD_COLS:
            std_norm = normalize_name(std)
            if std_norm == norm or std_norm in norm or norm in std_norm:
                matched = std
                break
        if matched and matched not in used_targets:
            mapping[col] = matched
            used_targets.add(matched)
            continue

        # sample-based inference
        infer = infer_by_samples(df[col])
        if infer and infer in STANDARD_COLS and infer not in used_targets:
            mapping[col] = infer
            used_targets.add(infer)
            continue

    # 4) final pass: if still missing Total Amount but Price & Quantity present, create mapping suggestion
    if "Total Amount" not in used_targets and "Price per Unit" in df.columns and "Quantity" in df.columns:
        # nothing to do; downstream handle calculated Total Amount
        pass

    return mapping


# ---------------- Apply mapping and cleaning ----------------
def clean_column_names(df: pd.DataFrame, prefer_llm=True) -> pd.DataFrame:
    """Map and rename columns to STANDARD_COLS using hybrid method."""
    mapping = map_columns_hybrid(df, prefer_llm=prefer_llm)
    if mapping:
        print(f"🔁 Applying column mapping: {mapping}")
        df = df.rename(columns=mapping)
    else:
        print("⚠️ No automatic mapping found - columns left unchanged.")
    return df


def clean_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to correct dtypes expected by analysis.py."""
    # numeric
    for col in ["Quantity", "Price per Unit", "Total Amount", "Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill / calculate common missing values and drop invalid rows."""
    # remove invalid quantities
    if "Quantity" in df.columns:
        df = df[df["Quantity"].notnull() & (df["Quantity"] > 0)]

    # Case 1: Total Amount missing, but Quantity and Price per Unit exist
    if "Total Amount" not in df.columns and "Price per Unit" in df.columns and "Quantity" in df.columns:
        df["Total Amount"] = df["Quantity"] * df["Price per Unit"]

    # Case 2: Total Amount exists but has NaN
    if "Total Amount" in df.columns:
        if "Price per Unit" in df.columns and "Quantity" in df.columns:
            mask = df["Total Amount"].isna() & df["Quantity"].notna() & df["Price per Unit"].notna()
            if mask.any():
                df.loc[mask, "Total Amount"] = df.loc[mask, "Quantity"] * df["Price per Unit"]

        # fill remaining NaNs with mean
        df["Total Amount"] = df["Total Amount"].fillna(df["Total Amount"].mean())

    # basic fill for Gender/Category
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].fillna("Unknown")

    if "Product Category" in df.columns:
        df["Product Category"] = df["Product Category"].fillna("Unknown")

    return df



def validate_required_columns(df: pd.DataFrame):
    """Ensure minimum columns for analysis are present - raise if critical missing."""
    required= [
    "Transaction ID", "Product Category","Total Amount"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for analysis: {missing}")
    return True


def clean_dataframe(df: pd.DataFrame, prefer_llm=True) -> pd.DataFrame:
    """Full cleaning pipeline: map columns, convert types, handle missing, validate."""
    print(f"📊 Starting cleanup - initial shape: {df.shape}")
    # 1. map/rename columns
    df = clean_column_names(df, prefer_llm=prefer_llm)
    # 2. convert types
    df = clean_data_types(df)
    # 3. handle missing / calculated values
    df = handle_missing_values(df)
    # 4. drop any fully empty rows
    initial = len(df)
    df = df.dropna(how='any')
    final = len(df)
    if initial > final:
        print(f"🧹 Dropped {initial - final} rows with remaining nulls")
    # 5. validate
    validate_required_columns(df)
    print(f"✅ Cleanup complete - final shape: {df.shape}")
    return df
