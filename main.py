# import visualization as vv
import pandas as pd
import analysis as ed

df = pd.read_csv('D:/data science/Projects/Business EDA/datasets/retail_sales_dataset.csv')

ans = ed.top_customers_by_sales(df)
print(ans)


data = [[1, 100], [2, 200], [3, 300]]
employee = pd.DataFrame(data, columns=['Id', 'Salary']).astype({'Id':'Int64', 'Salary':'Int64'})

print(employee)

uniquesal= employee['Salary'].drop_duplicates().sort_values(ascending=False)

print(uniquesal)
n = 1
data = {'SecondHighestSalary':[uniquesal.iloc[-1]]}

df= pd.DataFrame(data)
print(df)