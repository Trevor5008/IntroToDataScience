"""
Activities for Chapter 2 - Data Science Tools
The questions can be found in Canvas under:
    Chapter 2 - Gentle Introduction to Python Programming and Data Science Tools
"""

import pandas as pd

# Activity 1

products = pd.read_csv('datasets/products.csv')

print(products.head())
print(products.tail())
print(products.dtypes)
print(products.shape)
print(products.describe())

# Activity 2

sales_data = pd.read_csv('datasets/sales_data.csv')

print(sales_data.isnull().sum()) # Missing values

sales_data['Sales'] = sales_data['Sales'].fillna(sales_data['Sales'].mean())

sales_data['Date'] = pd.to_datetime(sales_data['Date'])

print(sales_data) # Cleaned dataset

# Activity 3

orders = pd.read_csv('datasets/orders.csv')

category_sales = orders.groupby('Category')['Amount'].sum()
print(category_sales) # Total sales by category

customer_avg = orders.groupby('Customer')['Amount'].mean()
print(customer_avg) # Average order amount by customer

# Activity 4

students = pd.read_csv("datasets/students.csv")
print(students.info())
students["Average"] = students[["Math", "Science", "English"]].mean(axis=1)
print(students[students["Average"]>85])

print(students["Average"].rank(ascending=False))