import matplotlib.pyplot as plt
import seaborn as sns
import analysis as an
import pandas as pd
import numpy as np

def visualize_top_selling_products(df):
    ans = an.top_selling_products(df)   
    axis = list(ans.index)
    values = list(ans.values)
    
    plt.figure(figsize=(8,5))
    x_pos = np.arange(len(axis))
    plt.bar(x_pos, values, color='skyblue')
    plt.xticks(x_pos, axis, rotation=45, ha='right')
    plt.xlabel('Product Category')
    plt.ylabel('Total Quantity Sold')
    plt.title('Top Selling Products')
    plt.tight_layout()
    return plt.show()

def visualize_top_selling_products_by_amount(df,n=10):
    ans = an.product_amount_wise_sales(df,n)
    axis = list(ans.index)
    values = list(ans.values)
    
    plt.figure(figsize=(8,5))
    x_pos = np.arange(len(axis))
    plt.bar(x_pos, values, color='skyblue')
    plt.xticks(x_pos, axis, rotation=45, ha='right')
    plt.xlabel('Product Category')
    plt.ylabel('Total Amount Sold')
    plt.title('Top Selling Products by Amount')
    plt.tight_layout() 
    plt.show()

def visualize_daily_sales_trend(df):
    ans = an.daily_sales_trend(df)
    axis = list(ans.index)
    values = list(ans.values)
    
    plt.figure(figsize=(12,8))
    sns.lineplot(x=axis, y=values)
    plt.title('Daily Sales Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.show()

def visualize_monthly_sales_trend(df):
    ans = an.monthly_sales_trend(df)
    
    plt.figure(figsize=(12,8))
    sns.lineplot(x='Month', y='Total Amount', data=ans)
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.show()

def visualize_genderwise_quantity(df):
    df = an.genderwise_quantity_table(df)
    df = df.T
    df.plot(kind='bar', figsize=(10,6), color=['hotpink','skyblue'])
    plt.xlabel('Product categories')
    plt.ylabel('Quantity Sold')
    plt.title('Gender wise Quantity Sold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

def visualize_genderwise_amount(df):
    df = an.genderwise_amount_table(df)
    df = df.T
    df.plot(kind='bar', figsize=(10,6), color=['hotpink','skyblue'])
    plt.xlabel('Product categories')
    plt.ylabel('Total Amount')
    plt.title('Gender wise Amount Sold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
    
def visualize_grouped_by_Age_group(df):
    df = an.age_group_analysis(df)
    pivot = df.pivot(index='Age Group', columns='Gender', values='Quantity').fillna(0)
    pivot.plot(kind='bar', figsize=(10,6), color=['hotpink','skyblue'])
    plt.xlabel('Age Group')
    plt.ylabel('Quantity Sold')
    plt.title('Age Group wise Quantity Sold')
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.show()
    
def visualize_grouped_by_Age_group_amount(df):
    dataf = an.age_group_analysis(df)
    pivot = dataf.pivot(index='Age Group', columns='Gender', values='Total Amount').fillna(0)
    pivot.plot(kind='bar', figsize=(10,6), color=['hotpink','skyblue'])
    plt.xlabel('Age Group')
    plt.ylabel('Total Amount')
    plt.title('Age Group wise Amount Sold')
    plt.xticks(rotation=45)
    plt.legend(title='Gender')
    plt.show()

def visualize_top_selling_days(df, n):
    dataf = an.top_selling_days(df, n)
    axis = list(dataf.index)
    values = list(dataf.values)

    plt.figure(figsize=(10,6))
    x_pos = np.arange(len(axis))
    plt.bar(x_pos, values, color='red')
    plt.xticks(x_pos, axis, rotation=45, ha='right')
    plt.xlabel('Days')
    plt.ylabel('Quantity Sold')
    plt.title(f'Top {n} Selling Days')
    plt.tight_layout()
    plt.show()
