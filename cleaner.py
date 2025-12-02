
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

try:
    import llmutil as ai
    LLM_AVAILABLE = True
except ImportError:
    print("Warning: llmutil module not available. Using basic column mapping only.")
    LLM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)   #create a reusable logging object

class DataCleanerConfig:
    """Configuration class for customizable cleaning parameters."""
    
    def __init__(self):
        # Standard columns expected by analysis.py
        self.STANDARD_COLS = [
            "Transaction ID", "Date", "Customer ID", "Gender", "Age",
            "Product Category", "Quantity", "Price per Unit", "Total Amount"
        ]
        
        # Validation bounds
        self.MIN_AGE = 0
        self.MAX_AGE = 100
        self.MIN_PRICE = 0.01
        self.MAX_PRICE = 1000000 
        self.MIN_QUANTITY = 0.01
        self.MAX_QUANTITY = 10000
        
        # Date validation
        self.MIN_DATE = datetime(1990, 1, 1)
        self.MAX_DATE = datetime.now() + timedelta(days=365) 
        
        # Missing value strategies
        self.CATEGORICAL_FILL = "Unknown"
        self.NUMERIC_FILL_STRATEGY = "median"  
        
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging/serialization."""
        return {
            "standard_cols": self.STANDARD_COLS,
            "age_bounds": (self.MIN_AGE, self.MAX_AGE),
            "price_bounds": (self.MIN_PRICE, self.MAX_PRICE),
            "quantity_bounds": (self.MIN_QUANTITY, self.MAX_QUANTITY),
            "date_bounds": (self.MIN_DATE.isoformat(), self.MAX_DATE.isoformat()),
            "fill_strategies": {
                "categorical": self.CATEGORICAL_FILL,
                "numeric": self.NUMERIC_FILL_STRATEGY
            }
        }

class DataCleaningStats:
    """Track cleaning statistics for reporting."""
    
    def __init__(self):
        self.initial_shape = None
        self.final_shape = None
        self.column_mappings = {}
        self.rows_removed = {}
        self.values_filled = {}
        self.values_calculated = {}
        self.validation_issues = {}
        
    def log_summary(self):
        """Print comprehensive cleaning summary."""
        logger.info("=" * 50)
        logger.info("DATA CLEANING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Initial shape: {self.initial_shape}")
        logger.info(f"Final shape: {self.final_shape}")
        
        if self.column_mappings:
            logger.info(f"Column mappings applied: {self.column_mappings}")
        
        if self.rows_removed:
            total_removed = sum(self.rows_removed.values())
            logger.info(f"Total rows removed: {total_removed}")
            for reason, count in self.rows_removed.items():
                logger.info(f"  - {reason}: {count}")
        
        if self.values_filled:
            total_filled = sum(self.values_filled.values())
            logger.info(f"Total values filled: {total_filled}")
            for column, count in self.values_filled.items():
                logger.info(f"  - {column}: {count}")
        
        if self.values_calculated:
            total_calc = sum(self.values_calculated.values())
            logger.info(f"Total values calculated: {total_calc}")
        
        if self.validation_issues:
            logger.warning("Validation issues found:")
            for issue, count in self.validation_issues.items():
                logger.warning(f"  - {issue}: {count}")

def finding_missing_columns(standard_col, column_list):
    """Find columns that are in standard_col but not in column_list."""
    missing_cols = list(set(standard_col) - set(column_list))
    return missing_cols


def validate_age(df: pd.DataFrame, config: DataCleanerConfig, stats: DataCleaningStats) -> pd.DataFrame:
    """Validate age values within reasonable bounds."""
    if "Age" not in df.columns:
        return df
    
    original_count = len(df)
    invalid_age_mask = (df["Age"] < config.MIN_AGE) | (df["Age"] > config.MAX_AGE)
    
    if invalid_age_mask.any():
        invalid_count = invalid_age_mask.sum()
        logger.warning(f"Found {invalid_count} ages outside bounds ({config.MIN_AGE}-{config.MAX_AGE})")
        stats.validation_issues["invalid_ages"] = invalid_count
        
        # Set invalid ages to NaN (will be filled later)
        df.loc[invalid_age_mask, "Age"] = np.nan
    
    return df

def validate_prices(df: pd.DataFrame, config: DataCleanerConfig, stats: DataCleaningStats) -> pd.DataFrame:
    """Validate price values within reasonable bounds."""
    if "Price per Unit" not in df.columns:
        return df
    
    invalid_price_mask = (df["Price per Unit"] <= 0) | (df["Price per Unit"] > config.MAX_PRICE)
    
    if invalid_price_mask.any():
        invalid_count = invalid_price_mask.sum()
        logger.warning(f"Found {invalid_count} prices outside bounds (${config.MIN_PRICE}-${config.MAX_PRICE})")
        stats.validation_issues["invalid_prices"] = invalid_count
        
        # Set invalid prices to NaN
        df.loc[invalid_price_mask, "Price per Unit"] = np.nan
    
    return df

def validate_quantities(df: pd.DataFrame, config: DataCleanerConfig, stats: DataCleaningStats) -> pd.DataFrame:
    """Validate quantity values within reasonable bounds."""
    if "Quantity" not in df.columns:
        return df
    
    invalid_qty_mask = (df["Quantity"] <= 0) | (df["Quantity"] > config.MAX_QUANTITY)
    
    if invalid_qty_mask.any():
        invalid_count = invalid_qty_mask.sum()
        logger.warning(f"Found {invalid_count} quantities outside bounds ({config.MIN_QUANTITY}-{config.MAX_QUANTITY})")
        stats.validation_issues["invalid_quantities"] = invalid_count
        
        # Remove rows with invalid quantities (these are critical)
        df = df[~invalid_qty_mask]
        stats.rows_removed["invalid_quantities"] = invalid_count
    
    return df

def validate_dates(df: pd.DataFrame, config: DataCleanerConfig, stats: DataCleaningStats) -> pd.DataFrame:
    """Validate date values within reasonable range."""
    if "Date" not in df.columns:
        return df
    
    invalid_date_mask = (df["Date"] < config.MIN_DATE) | (df["Date"] > config.MAX_DATE)
    
    if invalid_date_mask.any():
        invalid_count = invalid_date_mask.sum()
        logger.warning(f"Found {invalid_count} dates outside range ({config.MIN_DATE.date()}-{config.MAX_DATE.date()})")
        stats.validation_issues["invalid_dates"] = invalid_count
        
        # Set invalid dates to NaT
        df.loc[invalid_date_mask, "Date"] = pd.NaT
    
    return df
def validate_amount(df:pd.DataFrame,config:DataCleanerConfig,stats:DataCleaningStats)->pd.DataFrame:
    if "Total Amount" in df.columns:
        return df
    if ['Price per Unit','Quantity'] in df.columns:
        df['Total Amount']=df['Price per Unit']*df['Quantity']
        return df
    return df
def check_transaction_id_uniqueness(df: pd.DataFrame, stats: DataCleaningStats) -> pd.DataFrame:
    """Check for and handle duplicate transaction IDs."""
    if "Transaction ID" not in df.columns:
        return df
    
    initial_count = len(df)
    duplicate_mask = df["Transaction ID"].duplicated()
    
    if duplicate_mask.any():
        duplicate_count = duplicate_mask.sum()
        logger.warning(f"Found {duplicate_count} duplicate transaction IDs")
        stats.validation_issues["duplicate_transaction_ids"] = duplicate_count
        
        # Keep first occurrence, remove duplicates
        df = df[~duplicate_mask]
        stats.rows_removed["duplicate_transactions"] = duplicate_count
        logger.info(f"Removed {duplicate_count} duplicate transactions, keeping first occurrence")
    
    return df

def clean_column_names(df: pd.DataFrame, config: DataCleanerConfig, stats: DataCleaningStats, prefer_llm=True):
    """Map and rename columns to STANDARD_COLS using hybrid method."""
    # Find columns that need mapping
    current_cols = list(df.columns)
    missing_standard_cols = finding_missing_columns(config.STANDARD_COLS, current_cols)
    
    # Get columns that aren't already standard
    non_standard_cols = [col for col in current_cols if col not in config.STANDARD_COLS]
    
    mapping = {}
    
    if prefer_llm and LLM_AVAILABLE and missing_standard_cols and non_standard_cols:
        try:
            # Use LLM to infer mappings for missing columns
            inferred_cols = ai.infer_columns_with_gemini(non_standard_cols, missing_standard_cols)
            mapping = dict(zip(non_standard_cols, inferred_cols))
            logger.info("Using LLM-based column mapping")
        except Exception as e:
            logger.warning(f"LLM mapping failed: {e}. Using basic mapping.")
    
    # Apply the mapping
    if mapping:
        df = df.rename(columns=mapping)
        stats.column_mappings = mapping
        logger.info(f"Applied column mapping: {mapping}")

    # Add missing standard columns (filled with None)
    for col in config.STANDARD_COLS:
        if col not in df.columns:
            df[col] = None
            logger.info(f"Added missing column: {col}")
    
    # Reorder to match STANDARD_COLS and remove any extra columns
    df = df[config.STANDARD_COLS]

    return df

def clean_data_types(df: pd.DataFrame, stats: DataCleaningStats) -> pd.DataFrame:
    """Convert columns to correct dtypes expected by analysis.py."""
    logger.info("Converting data types...")
    
    # Convert numeric columns
    numeric_cols = ["Quantity", "Price per Unit", "Total Amount", "Age"]
    for col in numeric_cols:
        if col in df.columns:
            original_nulls = df[col].isnull().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            new_nulls = df[col].isnull().sum()
            if new_nulls > original_nulls:
                conversion_issues = new_nulls - original_nulls
                logger.warning(f"{col}: {conversion_issues} values converted to NaN")
                stats.validation_issues[f"{col}_conversion_errors"] = conversion_issues

    # Convert date column
    if "Date" in df.columns:
        original_nulls = df["Date"].isnull().sum()
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        new_nulls = df["Date"].isnull().sum()
        if new_nulls > original_nulls:
            conversion_issues = new_nulls - original_nulls
            logger.warning(f"Date: {conversion_issues} values converted to NaT")
            stats.validation_issues["date_conversion_errors"] = conversion_issues

    return df

def handle_missing_values(df: pd.DataFrame, config: DataCleanerConfig, stats: DataCleaningStats) -> pd.DataFrame:
    """Fill / calculate common missing values with configurable strategies."""
    logger.info("Handling missing values...")
    
    # Calculate missing Total Amount if possible
    if all(col in df.columns for col in ["Total Amount", "Price per Unit", "Quantity"]):
        calc_mask = (df["Total Amount"].isna() & 
                    df["Quantity"].notna() & 
                    df["Price per Unit"].notna())
        
        if calc_mask.any():
            calculated_count = calc_mask.sum()
            df.loc[calc_mask, "Total Amount"] = (df.loc[calc_mask, "Quantity"] * 
                                               df.loc[calc_mask, "Price per Unit"])
            logger.info(f"Calculated {calculated_count} missing Total Amount values")
            stats.values_calculated["Total Amount"] = calculated_count

    # Fill numeric columns based on strategy
    numeric_cols = ["Total Amount", "Price per Unit", "Quantity", "Age"]
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            na_count = df[col].isna().sum()
            
            if config.NUMERIC_FILL_STRATEGY == "median":
                fill_value = df[col].median()
            elif config.NUMERIC_FILL_STRATEGY == "mean":
                fill_value = df[col].mean()
            else:
                fill_value = config.NUMERIC_FILL_STRATEGY  # Assume it's a specific value
            
            df[col] = df[col].fillna(fill_value)
            logger.info(f"Filled {na_count} {col} NaNs with {config.NUMERIC_FILL_STRATEGY}: {fill_value:.2f}")
            stats.values_filled[col] = na_count

    # Fill categorical columns
    categorical_cols = ["Gender", "Product Category"]
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            na_count = df[col].isna().sum()
            df[col] = df[col].fillna(config.CATEGORICAL_FILL)
            logger.info(f"Filled {na_count} {col} NaNs with '{config.CATEGORICAL_FILL}'")
            stats.values_filled[col] = na_count

    return df

def validate_required_columns(df: pd.DataFrame, config: DataCleanerConfig):
    """Ensure minimum columns for analysis are present - raise if critical missing."""
    required = ["Transaction ID", "Product Category", "Total Amount"]
    
    missing = []
    for col in required:
        if col not in df.columns or df[col].isnull().all():
            missing.append(col)
    
    if missing:
        raise ValueError(f"Missing or empty required columns for analysis: {missing}")
    
    logger.info("All required columns validated")
    return True

def clean_dataframe(df: pd.DataFrame, config: Optional[DataCleanerConfig] = None, prefer_llm=True) -> Tuple[pd.DataFrame, DataCleaningStats]:
    """
    Full cleaning pipeline: map columns, convert types, handle missing, validate.
    
    Args:
        df: Input DataFrame
        config: DataCleanerConfig object (uses default if None)
        prefer_llm: Whether to prefer LLM-based column mapping
    
    Returns:
        Tuple of (cleaned_df, cleaning_stats)
    """
    if config is None:
        config = DataCleanerConfig()
    
    stats = DataCleaningStats()
    stats.initial_shape = df.shape
    
    logger.info(f"Starting cleanup - initial shape: {df.shape}")
    logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2, default=str)}")
    
    try:
        # 1. Map/rename columns
        df = clean_column_names(df, config, stats, prefer_llm=prefer_llm)
        
        # 2. Convert data types
        df = clean_data_types(df, stats)
        
        # 3. Validate data ranges
        df = validate_age(df, config, stats)
        df = validate_prices(df, config, stats)
        df = validate_quantities(df, config, stats)
        df = validate_dates(df, config, stats)
        df= validate_amount(df,config,stats)
        # 4. Check transaction ID uniqueness
        df = check_transaction_id_uniqueness(df, stats)
        
        # 5. Handle missing/calculated values
        df = handle_missing_values(df, config, stats)
        
        # 6. Drop rows that are still problematic
        initial_count = len(df)
        
        # Keep rows that have at least Transaction ID and Total Amount
        essential_mask = (df["Transaction ID"].notnull() & 
                         df["Total Amount"].notnull())
        df = df[essential_mask]
        
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows missing essential data")
            stats.rows_removed["missing_essential_data"] = dropped
        
        # 7. Validate final result
        # validate_required_columns(df, config)
        
        stats.final_shape = df.shape
        stats.log_summary()
        
        return df, stats
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise

# Convenience function for backward compatibility
def clean_dataframe_simple(df: pd.DataFrame, prefer_llm=True) -> pd.DataFrame:
    """Simple wrapper that returns only the cleaned DataFrame (original behavior)."""
    cleaned_df, _ = clean_dataframe(df, prefer_llm=prefer_llm)
    return cleaned_df

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use with custom configuration
    custom_config = DataCleanerConfig()
    custom_config.MAX_AGE = 100  # Lower age limit
    custom_config.CATEGORICAL_FILL = "N/A"  # Different fill value
    
    # Example DataFrame
    sample_data = {
        'txn_id': ['T001', 'T002', 'T003'],
        'purchase_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'customer_gender': ['M', 'F', None],
        'customer_age': [25, 150, 30],  # 150 is invalid
        'product_type': ['Electronics', 'Clothing', 'Books'],
        'qty': [2, 1, 3],
        'unit_price': [100.0, 50.0, 15.0],
        'total_cost': [200.0, None, 45.0]  # One missing value
    }
    
    df = pd.read_csv('D:/data science/Projects/Business EDA/datasets/train.csv')    
    print("Original DataFrame:")
    print(df)
    
    # Clean with custom config
    cleaned_df, cleaning_stats = clean_dataframe(df, config=custom_config, prefer_llm=False)
    print("\nCleaned DataFrame:")
    print(cleaned_df)