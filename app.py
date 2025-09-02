import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io

# Import your modules
try:
    import cleaner
    import analysis as an
    import visualization as viz
    import llmutil as ai
except ImportError as e:
    st.error(f"Import error: {e}")

st.set_page_config(
    page_title="Business Data Analyzer",
    page_icon="ðŸ“Š",
    layout="wide"
)

def load_file_with_encoding(uploaded_file):
    """Load CSV or Excel with multiple encoding attempts"""
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Handle Excel files
    if file_extension in ['xlsx', 'xls']:
        try:
            uploaded_file.seek(0)
            # Try to read Excel file
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
            st.success(f"âœ… Excel file loaded successfully")
            return df
        except Exception as e:
            st.error(f"âŒ Failed to load Excel file: {e}")
            return None
    
    # Handle CSV files
    elif file_extension == 'csv':
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                st.success(f"âœ… CSV file loaded with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings[-1]:
                    st.error(f"âŒ Failed to load CSV file: {e}")
                    return None
    
    else:
        st.error(f"âŒ Unsupported file format: {file_extension}")
        return None
    
    st.error("âŒ Could not load file")
    return None

def setup_api():
    """Setup Gemini API"""
    
    if 'api_ready' not in st.session_state:
        st.session_state.api_ready = False
    
    if not st.session_state.api_ready:
        with st.sidebar:
            st.header("ðŸ”‘ API Setup")
            api_key = st.text_input("Gemini API Key:", type="password")
            
            if api_key:
                try:
                    ai.setup_gemini(api_key)
                    st.session_state.api_ready = True
                    st.success("âœ… API configured!")
                    st.rerun()
                except Exception as e:
                    st.error(f"âŒ API error: {e}")
            else:
                st.info("Enter your API key to continue")
                return False
    
    return True

def run_analysis(df):
    """Run all analysis functions and return results"""
    
    results = {}
    
    # Basic summary
    results['summary'] = an.total_sales_summary(df)
    
    # Product analysis
    results['top_products_qty'] = an.top_selling_products(df).to_dict()
    results['top_products_revenue'] = an.product_amount_wise_sales(df).to_dict()
    
    # Trends
    results['monthly_trend'] = an.monthly_sales_trend(df).to_dict('records')
    
    # Demographics
    if 'Gender' in df.columns:
        results['age_analysis'] = an.age_group_analysis(df).to_dict('records')
    
    # Top performers
    results['top_days'] = an.top_selling_days(df, 5).to_dict()
    results['top_customers'] = an.top_customers_by_sales(df, 5).to_dict()
    results['avg_basket'] = an.average_basket_size(df)
    
    return results

def create_visualizations(df):
    """Create and display all visualizations"""
    
    # Products
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Products by Quantity")
        viz.visualize_top_selling_products(df)
        st.pyplot(plt.gcf())
        plt.close()
    
    with col2:
        st.subheader("Top Products by Revenue")
        viz.visualize_top_selling_products_by_amount(df)
        st.pyplot(plt.gcf())
        plt.close()
    
    # Trends
    st.subheader("Sales Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Daily Trend**")
        viz.visualize_daily_sales_trend(df)
        st.pyplot(plt.gcf())
        plt.close()
    
    with col2:
        st.write("**Monthly Trend**")
        viz.visualize_monthly_sales_trend(df)
        st.pyplot(plt.gcf())
        plt.close()
    
    # Demographics (if available)
    if 'Gender' in df.columns:
        st.subheader("Demographics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Gender-wise Quantity**")
            viz.visualize_genderwise_quantity(df)
            st.pyplot(plt.gcf())
            plt.close()
        
        with col2:
            st.write("**Age Group Analysis**")
            viz.visualize_grouped_by_Age_group(df)
            st.pyplot(plt.gcf())
            plt.close()

def main():
    st.title("ðŸ“Š Business Data Analyzer")
    st.markdown("Upload your CSV and get AI-powered insights!")
    
    # Setup API
    if not setup_api():
        st.warning("âš ï¸ Please configure your API key first")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your file", 
        type=["csv", "xlsx", "xls"],
        help="Upload business data in CSV or Excel format - columns will be automatically standardized"
    )
    
    if uploaded_file is not None:
        
        # Load file
        df = load_file_with_encoding(uploaded_file)
        if df is None:
            return
        
        st.success(f"ðŸ“ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show original data
        with st.expander("ðŸ“‹ Original Data"):
            st.write("**Columns:**", list(df.columns))
            st.dataframe(df.head())
        
        # Clean data
        try:
            with st.spinner("ðŸ§¹ Cleaning data with AI..."):
                cleaned_df = cleaner.clean_dataframe(df.copy())
            
            st.success(f"âœ… Data cleaned! Final shape: {cleaned_df.shape}")
            
            # Show mapping debug info
            with st.expander("ðŸ” Column Mapping Debug"):
                st.write("**Original â†’** **Standardized**")
                original_cols = set(df.columns)
                cleaned_cols = set(cleaned_df.columns)
                
                for orig_col in df.columns:
                    if orig_col in cleaned_df.columns:
                        st.write(f"â€¢ {orig_col} â†’ {orig_col} (unchanged)")
                    else:
                        # Find what it was mapped to
                        for new_col in cleaned_cols:
                            if new_col not in original_cols:
                                st.write(f"â€¢ {orig_col} â†’ **{new_col}**")
                                break
            
        except Exception as e:
            st.error(f"âŒ Cleaning failed: {str(e)}")
            st.write("**Available columns:**", list(df.columns))
            
            # Show what we can try to map manually
            st.write("**Try manual mapping:**")
            fallback_mapping = cleaner.smart_fallback_mapping(df.columns)
            if fallback_mapping:
                st.write("**Suggested mappings:**", fallback_mapping)
            else:
                st.write("No automatic mappings found. Please check your column names.")
            return
        
        # Show cleaned data
        with st.expander("âœ¨ Cleaned Data"):
            st.write("**Standardized columns:**", list(cleaned_df.columns))
            st.dataframe(cleaned_df.head())
        
        # Analysis
        st.header("ðŸ“ˆ Analysis Results")
        
        # Key metrics
        summary = an.total_sales_summary(cleaned_df)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ðŸ’° Total Sales", f"${summary['Total sales']:,.2f}")
        with col2:
            st.metric("ðŸ“¦ Quantity Sold", f"{summary['Total quantities sold']:,}")
        with col3:
            st.metric("ðŸ›’ Transactions", f"{summary['Total Transaction']:,}")
        
        # Visualizations
        st.subheader("ðŸ“Š Visualizations")
        create_visualizations(cleaned_df)
        
        # AI Summary
        st.header("ðŸ¤– AI Business Summary")
        
        with st.spinner("Generating AI insights..."):
            analysis_results = run_analysis(cleaned_df)
            
            df_info = {
                'rows': len(cleaned_df),
                'date_range': f"{cleaned_df['Date'].min()} to {cleaned_df['Date'].max()}" if 'Date' in cleaned_df.columns else 'N/A'
            }
            
            summary_text = ai.generate_summary(analysis_results, df_info)
        
        st.markdown(summary_text)
        
        # Download report
        st.download_button(
            label="ðŸ“„ Download Report",
            data=summary_text,
            file_name=f"business_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()