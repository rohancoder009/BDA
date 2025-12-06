import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import analysis as an

plt.style.use("default")
sns.set_palette("husl")

# Helper for charts
def _fig_ax(figsize=(4, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

# ======================================================
# BASIC EXISTING VISUALIZATIONS
# ======================================================
def visualize_top_selling_products(df, n=10):
    data = an.top_selling_products(df, n)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values)
    ax.set_title("Top Selling Products (Quantity)")
    ax.set_ylabel("Units Sold")
    plt.xticks(rotation=45)
    return fig


def visualize_top_selling_products_by_amount(df, n=10):
    data = an.product_amount_wise_sales(df, n)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values, color="orange")
    ax.set_title("Top Selling Products by Revenue")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)
    return fig


def visualize_daily_sales_trend(df):
    data = an.daily_sales_trend(df)
    fig, ax = _fig_ax((14, 6))

    ax.plot(data.index, data.values, marker="o")
    ax.fill_between(data.index, data.values, alpha=0.25)
    ax.set_title("Daily Sales Trend")
    plt.xticks(rotation=45)
    return fig


def visualize_monthly_sales_trend(df):
    data = an.monthly_sales_trend(df)
    fig, ax = _fig_ax((14, 6))

    ax.plot(data["Month"], data["Total Amount"], marker="o", linewidth=3)
    ax.set_title("Monthly Sales Trend")
    plt.xticks(rotation=45)
    return fig

# ======================================================
# ADVANCED VISUALIZATION FOR ALL NEW FUNCTIONS
# ======================================================

# ------------------ WEEKLY SALES ----------------------
def visualize_weekly_sales(df):
    data = an.weekly_sales_summary(df)
    fig, ax = _fig_ax()

    ax.plot(data.index, data.values, marker="o")
    ax.set_title("Weekly Sales Summary")
    ax.set_ylabel("Revenue")
    return fig

# ------------------ GROWTH RATE -----------------------
def visualize_monthly_growth_rate(df):
    data = an.monthly_growth_rate(df)
    fig, ax = _fig_ax()

    ax.bar(data["Month"], data["Growth %"], color="green")
    ax.axhline(0, linestyle="--", color="red")
    ax.set_title("Month-over-Month Growth (%)")
    plt.xticks(rotation=45)
    return fig

# ------------------ SALES DISTRIBUTION ----------------
def visualize_sales_distribution(df):
    data = an.sales_distribution_by_category(df)
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.pie(data, labels=data.index, autopct="%1.1f%%")
    ax.set_title("Sales Distribution by Category")
    return fig

# ------------------ PARETO ----------------------------
def visualize_revenue_pareto(df):
    data = an.revenue_contribution(df)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values)
    ax.set_title("Pareto Analysis (Cumulative Revenue %)")
    plt.xticks(rotation=45)
    return fig

# ------------------ SLOW PRODUCTS ---------------------
def visualize_slow_moving_products(df, n=5):
    data = an.slow_moving_products(df, n)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values, color="red")
    ax.set_title("Slow Moving Products")
    plt.xticks(rotation=45)
    return fig

# ------------------ FAST PRODUCTS ---------------------
def visualize_fast_moving_products(df, n=5):
    data = an.fast_moving_products(df, n)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values, color="green")
    ax.set_title("Fast Moving Products")
    plt.xticks(rotation=45)
    return fig

# ------------------ PRICE ELASTICITY -------------------
def visualize_price_elasticity(df):
    data = an.price_elasticity(df)
    fig, ax = _fig_ax()

    ax.scatter(data["avg_price"], data["total_qty"], s=120)
    ax.set_title("Price Elasticity (Price vs Quantity)")
    ax.set_xlabel("Avg Price")
    ax.set_ylabel("Total Quantity")
    return fig

# ------------------ PRODUCT MONTHLY PERFORMANCE --------
def visualize_product_monthly_performance(df):
    data = an.product_monthly_performance(df)
    fig, ax = _fig_ax((14, 6))

    for cat in data["Product Category"].unique():
        temp = data[data["Product Category"] == cat]
        ax.plot(temp["Month"].astype(str), temp["Total Amount"], marker="o", label=cat)



    ax.legend()
    ax.set_title("Product Monthly Revenue Performance")
    plt.xticks(rotation=45)
    return fig

# ------------------ NEW VS RETURNING -------------------
def visualize_new_vs_returning(df):
    data = an.new_vs_returning_customers(df)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values)
    ax.set_title("New vs Returning Customers")
    return fig

# ------------------ RETENTION RATE ---------------------
def visualize_retention_rate(df):
    data = an.customer_retention_rate(df)
    fig, ax = _fig_ax()

    ax.plot(data.index.astype(str), data.values, marker="o")
    ax.set_title("Customer Retention Rate (%)")
    plt.xticks(rotation=45)
    return fig

# ------------------ RFM HEATMAP ------------------------
def visualize_rfm_heatmap(df):
    rfm = an.rfm_segmentation(df)
    fig, ax = _fig_ax((12, 6))

    sns.heatmap(rfm.corr(), annot=True, cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("RFM Correlation Heatmap")
    return fig

# ------------------ COHORT ANALYSIS --------------------
def visualize_cohort_analysis(df):
    cohort = an.customer_cohort_analysis(df)
    pivot = cohort.pivot(index="Cohort Month", columns="Order Month", values="Customer ID")

    fig, ax = _fig_ax((12, 6))
    sns.heatmap(pivot, cmap="Greens", annot=True, fmt=".0f", ax=ax)
    ax.set_title("Customer Cohort Retention")
    return fig

# ------------------ SEASONAL PATTERN -------------------
def visualize_seasonal_sales(df):
    data = an.seasonal_sales_patterns(df)
    fig, ax = _fig_ax()

    ax.plot(data.index, data.values, marker="o")
    ax.set_title("Seasonal Sales Pattern")
    ax.set_xlabel("Month")
    return fig

# ------------------ MOVING AVERAGE ---------------------
def visualize_moving_average(df, window=7):
    data = an.moving_average_sales(df, window)
    fig, ax = _fig_ax((14, 6))

    ax.plot(data.index, data.values, marker="o")
    ax.set_title(f"{window}-Day Moving Average Trend")
    return fig

# ------------------ SALES FORECAST ---------------------
def visualize_sales_forecast(df, periods=12):
    forecast = an.sales_forecast(df, periods)
    fig, ax = _fig_ax()

    ax.plot(forecast.index, forecast.values, marker="o")
    ax.set_title("Simple Sales Forecast")
    return fig

# ------------------ ANOMALIES --------------------------
def visualize_sales_anomalies(df):
    anomalies = an.anomaly_detection(df)
    daily = an.daily_sales_trend(df)

    fig, ax = _fig_ax((14, 6))

    ax.plot(daily.index, daily.values, label="Sales")
    ax.scatter(anomalies.index, anomalies.values, color="red", s=100, label="Anomaly")
    ax.legend()
    ax.set_title("Sales Anomaly Detection")
    return fig

# ------------------ AGE GROUP --------------------------
def visualize_age_group_analysis(df):
    data = an.age_group_analysis(df)
    fig, ax = _fig_ax((12, 6))

    sns.barplot(data=data, x="Age Group", y="Total Amount", hue="Gender", ax=ax)
    ax.set_title("Age Group Spending Comparison")
    return fig

# ------------------ GENDER QUANTITY ---------------------
def visualize_genderwise_quantity(df):
    data = an.genderwise_quantity_table(df)
    fig, ax = _fig_ax((14, 6))

    sns.heatmap(data, annot=True, cmap="Purples", ax=ax)
    ax.set_title("Gender-wise Quantity Heatmap")
    return fig

# ------------------ GENDER REVENUE ----------------------
def visualize_genderwise_amount(df):
    data = an.genderwise_amount_table(df)
    fig, ax = _fig_ax((14, 6))

    sns.heatmap(data, annot=True, cmap="Oranges", ax=ax)
    ax.set_title("Gender-wise Revenue Heatmap")
    return fig

# ------------------ TOP SELLING DAYS --------------------
def visualize_top_selling_days(df, n=5):
    data = an.top_selling_days(df, n)
    fig, ax = _fig_ax((14, 6))

    ax.bar(data.index, data.values)
    ax.set_title("Top Selling Days")
    plt.xticks(rotation=45)
    return fig

# ------------------ TOP CUSTOMERS -----------------------
def visualize_top_customers(df, n=10):
    data = an.top_customers_by_sales(df, n)
    fig, ax = _fig_ax((12, 6))

    ax.bar(data.index, data.values)
    ax.set_title("Top Customers by Revenue")
    plt.xticks(rotation=45)
    return fig

# ------------------ PROFIT ------------------------------
def visualize_total_profit(df):
    if "Cost Price" not in df.columns:
        raise ValueError("Cost Price column missing")

    df["Profit"] = (df["Price per Unit"] - df["Cost Price"]) * df["Quantity"]
    data = df.groupby("Product Category")["Profit"].sum()

    fig, ax = _fig_ax((12, 6))
    ax.bar(data.index, data.values, color="green")
    ax.set_title("Profit by Product Category")
    plt.xticks(rotation=45)
    return fig

# ------------------ PROFIT MARGIN -----------------------
def visualize_profit_margin(df):
    if "Cost Price" not in df.columns:
        raise ValueError("Cost Price column missing")

    profit_margin = an.profit_margin(df)

    fig, ax = _fig_ax()
    ax.bar(["Profit Margin %"], [profit_margin], color="blue")
    ax.set_title("Overall Profit Margin")
    return fig

# ------------------ STOCK OUT RISK ----------------------
def visualize_stock_out_risk(df, inventory_df):
    data = an.stock_out_risk(df, inventory_df)
    fig, ax = _fig_ax((12, 6))

    ax.bar(data["Product Category"], data["Days Left"], color="red")
    ax.set_title("Stock-out Risk (Days Left)")
    plt.xticks(rotation=45)
    return fig

# ------------------ REORDER SUGGESTION ------------------
def visualize_reorder_suggestion(df, inventory_df):
    data = an.reorder_suggestion(df, inventory_df)
    fig, ax = _fig_ax((12, 6))

    ax.bar(data["Product Category"], data["Reorder Point"], color="purple")
    ax.set_title("Reorder Point Suggestion")
    plt.xticks(rotation=45)
    return fig

# ------------------ REGIONAL SALES ----------------------
def visualize_regional_sales(df):
    if "Region" not in df.columns:
        raise ValueError("Region column missing")

    data = an.regional_sales_analysis(df)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values)
    ax.set_title("Regional Revenue Comparison")
    plt.xticks(rotation=45)
    return fig

# ------------------ DISCOUNT EFFECTIVENESS --------------
def visualize_discount_effectiveness(df):
    if "Discount" not in df.columns:
        raise ValueError("Discount column missing")

    data = an.discount_effectiveness(df)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values)
    ax.set_title("Discount Effectiveness (Revenue by Discount Level)")
    return fig

# ------------------ REFUND RATE -------------------------
def visualize_refund_rate(df):
    if "Refund" not in df.columns:
        raise ValueError("Refund column missing")

    data = an.refund_rate(df)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values, color="red")
    ax.set_title("Refund Count by Category")
    return fig

# ------------------ PAYMENT METHODS ---------------------
def visualize_payment_methods(df):
    if "Payment Method" not in df.columns:
        raise ValueError("Payment Method column missing")

    data = an.top_payment_methods(df)
    fig, ax = _fig_ax()

    ax.bar(data.index, data.values)
    ax.set_title("Most Used Payment Methods")
    return fig
