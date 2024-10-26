import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create a connection to the database
conn = sqlite3.connect('RevenueData.db')

# Create customers table
conn.execute('''
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT,
    segment TEXT,
    region TEXT,
    join_date DATE
)
''')

# Create products table
conn.execute('''
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    unit_price DECIMAL(10,2)
)
''')

# Create sales table
conn.execute('''
CREATE TABLE sales (
    sale_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    sale_date DATE,
    quantity INTEGER,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
''')

# Generate sample customer data
customers_data = []
segments = ['Enterprise', 'SMB', 'Consumer']
regions = ['North', 'South', 'East', 'West']
start_date = datetime(2023, 1, 1)

for i in range(100):
    join_date = start_date + timedelta(days=np.random.randint(0, 365))
    customers_data.append({
        'customer_id': i + 1,
        'customer_name': f'Customer {i+1}',
        'segment': np.random.choice(segments),
        'region': np.random.choice(regions),
        'join_date': join_date.strftime('%Y-%m-%d')
    })

# Generate sample product data
products_data = []
categories = ['Software', 'Hardware', 'Services', 'Consulting']
base_prices = {'Software': 1000, 'Hardware': 500, 'Services': 200, 'Consulting': 2000}

for i in range(20):
    category = np.random.choice(categories)
    products_data.append({
        'product_id': i + 1,
        'product_name': f'Product {i+1}',
        'category': category,
        'unit_price': round(base_prices[category] * (0.8 + np.random.random() * 0.4), 2)
    })

# Generate sample sales data
sales_data = []
for i in range(1000):
    sale_date = start_date + timedelta(days=np.random.randint(0, 365))
    customer_id = np.random.randint(1, 101)
    product_id = np.random.randint(1, 21)
    quantity = np.random.randint(1, 10)
    
    # Get product price
    product_price = [p['unit_price'] for p in products_data if p['product_id'] == product_id][0]
    total_amount = round(quantity * product_price, 2)
    
    sales_data.append({
        'sale_id': i + 1,
        'customer_id': customer_id,
        'product_id': product_id,
        'sale_date': sale_date.strftime('%Y-%m-%d'),
        'quantity': quantity,
        'total_amount': total_amount
    })

# Convert to DataFrames and save to database
pd.DataFrame(customers_data).to_sql('customers', conn, if_exists='append', index=False)
pd.DataFrame(products_data).to_sql('products', conn, if_exists='append', index=False)
pd.DataFrame(sales_data).to_sql('sales', conn, if_exists='append', index=False)

conn.close()