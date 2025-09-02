import pandas as pd
import numpy as np




def total_sales_summary(df):
    total_sales = df['Total Amount'].sum()
    total_quantity = df['Quantity'].sum()
    total_transactions = df.shape[0]
    return {
        'Total sales' :total_sales,
        'Total quantities sold' : total_quantity,
        'Total Transaction':  total_transactions
    }

def top_selling_products(df, n=10):
    return df.groupby('Product Category')['Quantity'].sum().sort_values(ascending = False).head(n)

def product_amount_wise_sales(df,n=10):
    return df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending = False).head(n)

def daily_sales_trend(df):
    df['Date']=pd.to_datetime(df['Date'])
    return df.groupby('Date')['Total Amount'].sum()

def monthly_sales_trend(df):
    df['Date']=pd.to_datetime(df['Date'])
    df['Month']= df['Date'].dt.to_period('M')
    monthly = df.groupby('Month')['Total Amount'].sum().reset_index()
    monthly['Month'] = monthly['Month'].astype(str)
    return monthly


def genderwise_quantity_table(df):
    return df.pivot_table(index='Gender', columns='Product Category', values='Quantity', aggfunc='sum', fill_value=0)

def genderwise_amount_table(df):
    return df.pivot_table(index='Gender', columns='Product Category', values='Total Amount', aggfunc='sum', fill_value=0)

def customer_purchase_frequency(df):
    # number of highest order by a customer
    return df.groupby(['Customer ID', 'Gender'])[['Quantity', 'Total Amount']].sum().sort_values(by='Quantity', ascending=False).head(5)

def age_group_analysis(df):
    bins = [0,18,25,35,50,65,100]
    labels = ['TEEN','Youth','Young','adult','Middle age','senior']
    
    df['Age Group'] = pd.cut(df['Age'],bins=bins,labels=labels,right=False)
    grouped =df.groupby(['Gender','Age Group'])[['Quantity','Total Amount']].sum().reset_index()
    return grouped
def top_selling_days(df , n=5):
    df['Date'] = pd.to_datetime(df['Date'])
    return df.groupby('Date')['Total Amount'].sum().sort_values(ascending =False).head(n)

def top_customers_by_sales(df,n=5):
    return df.groupby(['Customer ID'])['Total Amount'].sum().sort_values(ascending = False).head(n)

def average_basket_size(df):
    return round(df['Quantity'].mean(),2)


