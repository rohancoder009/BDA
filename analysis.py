import pandas as pd
import numpy as np


def total_sales_summary(df):
    total_sales = df['Total Amount'].sum()
    total_quantity = df['Quantity'].sum()
    total_transactions = df.shape[0]
    return {
        'Total sales' : total_sales,
        'Total quantities sold' : total_quantity,
        'Total Transaction': total_transactions
    }

def top_selling_products(df, n=10):
    return df.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False).head(n)

def product_amount_wise_sales(df, n=10):
    return df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False).head(n)

def daily_sales_trend(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df.groupby('Date')['Total Amount'].sum()

def monthly_sales_trend(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly = df.groupby('Month')['Total Amount'].sum().reset_index()
    monthly['Month'] = monthly['Month'].astype(str)
    return monthly

def genderwise_quantity_table(df):
    return df.pivot_table(index='Gender', columns='Product Category', values='Quantity', aggfunc='sum', fill_value=0)

def genderwise_amount_table(df):
    return df.pivot_table(index='Gender', columns='Product Category', values='Total Amount', aggfunc='sum', fill_value=0)

def customer_purchase_frequency(df):
    return df.groupby(['Customer ID', 'Gender'])[['Quantity', 'Total Amount']].sum().sort_values(by='Quantity', ascending=False).head(5)

def age_group_analysis(df):
    bins = [0,18,25,35,50,65,100]
    labels = ['TEEN','Youth','Young','Adult','Middle Age','Senior']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    grouped = df.groupby(['Gender','Age Group'])[['Quantity','Total Amount']].sum().reset_index()
    return grouped

def top_selling_days(df, n=5):
    df['Date'] = pd.to_datetime(df['Date'])
    return df.groupby('Date')['Total Amount'].sum().sort_values(ascending=False).head(n)

def top_customers_by_sales(df, n=5):
    return df.groupby('Customer ID')['Total Amount'].sum().sort_values(ascending=False).head(n)

def average_basket_size(df):
    return round(df['Quantity'].mean(), 2)

# ---------- SALES ANALYSIS ----------

def weekly_sales_summary(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week'] = df['Date'].dt.isocalendar().week
    return df.groupby('Week')['Total Amount'].sum()

def monthly_growth_rate(df):
    monthly = monthly_sales_trend(df)
    monthly['Growth %'] = monthly['Total Amount'].pct_change() * 100
    return monthly

def average_order_value(df):
    return df['Total Amount'].sum() / df.shape[0]

def sales_distribution_by_category(df):
    total = df['Total Amount'].sum()
    category_share = df.groupby("Product Category")['Total Amount'].sum()
    return (category_share / total * 100).sort_values(ascending=False)

def revenue_contribution(df, threshold=80):
    contrib = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
    cumulative = contrib.cumsum() / contrib.sum() * 100
    return cumulative[cumulative <= threshold]

# ---------- PRODUCT ANALYSIS ----------

def slow_moving_products(df, n=5):
    prod = df.groupby('Product Category')['Quantity'].sum().sort_values()
    return prod.head(n)

def fast_moving_products(df, n=5):
    prod = df.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False)
    return prod.head(n)

def product_repeat_rate(df):
    repeat = df.groupby('Product Category')['Customer ID'].nunique()
    total_customers = df['Customer ID'].nunique()
    return (repeat / total_customers * 100).sort_values(ascending=False)

def price_elasticity(df):
    grouped = df.groupby(['Product Category']).agg(
        avg_price=('Price per Unit', 'mean'),
        total_qty=('Quantity', 'sum')
    )
    grouped['Elasticity'] = grouped['total_qty'].pct_change() / grouped['avg_price'].pct_change()
    return grouped

def product_monthly_performance(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    
    return df.groupby(['Product Category','Month'])[['Quantity','Total Amount']].sum().reset_index()

# ---------- CUSTOMER ANALYSIS ----------

def new_vs_returning_customers(df):
    first_purchase = df.groupby('Customer ID')['Date'].min()
    df = df.merge(first_purchase.rename('First Purchase Date'), on='Customer ID')
    df['Customer Type'] = np.where(df['Date'] == df['First Purchase Date'], 'New', 'Returning')
    return df['Customer Type'].value_counts()

def customer_retention_rate(df):
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly_customers = df.groupby('Month')['Customer ID'].nunique()
    retention = monthly_customers.pct_change() * 100
    return retention

def customer_lifetime_value(df):
    avg_order_value = df['Total Amount'].mean()
    frequency = df.groupby('Customer ID').size().mean()
    return avg_order_value * frequency

def rfm_segmentation(df):
    df['Date'] = pd.to_datetime(df['Date'])
    snapshot_date = df['Date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({
        'Date': lambda x: (snapshot_date - x.max()).days,
        'Customer ID': 'count',
        'Total Amount': 'sum'
    }).rename(columns={'Date': 'Recency', 'Customer ID': 'Frequency', 'Total Amount': 'Monetary'})

    return rfm.sort_values('Monetary', ascending=False)

def customer_cohort_analysis(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Order Month'] = df['Date'].dt.to_period('M')
    df['Cohort Month'] = df.groupby('Customer ID')['Order Month'].transform('min')
    cohort = df.groupby(['Cohort Month', 'Order Month'])['Customer ID'].nunique().reset_index()
    return cohort

# ---------- TREND & FORECASTING ----------

def seasonal_sales_patterns(df):
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    return df.groupby('Month')['Total Amount'].sum()

def moving_average_sales(df, window=7):
    daily = daily_sales_trend(df)
    return daily.rolling(window).mean()

def sales_forecast(df, periods=12):
    monthly = monthly_sales_trend(df)
    last_value = monthly['Total Amount'].iloc[-1]
    forecast = pd.Series([last_value] * periods)
    return forecast

def anomaly_detection(df):
    daily = daily_sales_trend(df)
    mean = daily.mean()
    std = daily.std()
    anomalies = daily[(daily > mean + 3*std) | (daily < mean - 3*std)]
    return anomalies

# ---------- PROFIT & OPERATIONS ----------

def total_profit(df):
    if 'CostPrice' in df.columns:
        df['Profit'] = (df['Price per Unit'] - df['Cost Price']) * df['Quantity']
        return df['Profit'].sum()
    return "Cost Price column missing"

def profit_margin(df):
    if 'CostPrice' in df.columns:
        profit = total_profit(df)
        revenue = df['Total Amount'].sum()
        return (profit / revenue) * 100
    return "Cost Price column missing"

def stock_out_risk(df, inventory_df):
    daily_sales = df.groupby('Product Category')['Quantity'].mean()
    inventory_df['Daily Avg Sales'] = inventory_df['Product Category'].map(daily_sales)
    inventory_df['Days Left'] = inventory_df['Stock'] / inventory_df['Daily Avg Sales']
    return inventory_df.sort_values('Days Left')

def reorder_suggestion(df, inventory_df, lead_time=7, safety_stock=20):
    daily_sales = df.groupby('Product Category')['Quantity'].mean()
    inventory_df['Daily Avg Sales'] = inventory_df['Product Category'].map(daily_sales)
    inventory_df['Reorder Point'] = (inventory_df['Daily Avg Sales'] * lead_time) + safety_stock
    return inventory_df[['Product Category', 'Reorder Point']]

def regional_sales_analysis(df):
    if 'Region' in df.columns:
        return df.groupby('Region')['Total Amount'].sum().sort_values(ascending=False)
    return "Region column not found"

def discount_effectiveness(df):
    if 'Discount' in df.columns:
        return df.groupby('Discount')['Total Amount'].sum()
    return "Discount column missing"

def refund_rate(df):
    if 'Refund' in df.columns:
        return df.groupby('Product Category')['Refund'].sum()
    return "Refund column missing"

def top_payment_methods(df):
    if 'Payment Method' in df.columns:
        return df['Payment Method'].value_counts()
    return "Payment Method column missing"
