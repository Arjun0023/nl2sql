import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create a connection to the database
conn = sqlite3.connect('TrainingData.db')

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

# Generate realistic company names
company_names = [
    "TechNova Inc.", "Greenfield Solutions", "AstroTech", "BlueSky Systems", "DeltaSoft",
    "NextGen Hardware", "Pinnacle Services", "CloudAxis", "SmartEdge Technologies", "Visionary Consulting",
    "PrimeWare", "Core Dynamics", "Skyline Enterprises", "EcoLogic Innovations", "Fusion Networks",
    "Vanguard Systems", "Infinity Solutions", "QuantumSoft", "Summit Technologies", "AgileWorks"
]

# Generate sample customer data
customers_data = []
segments = ['Enterprise', 'SMB', 'Consumer']
regions = ['North', 'South', 'East', 'West']
start_date = datetime(2021, 1, 1)  # Start date for join dates
end_date = datetime(2024, 12, 31)  # End date for join dates

for i, company_name in enumerate(company_names):
    join_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days + 1))
    customers_data.append({
        'customer_id': i + 1,
        'customer_name': company_name,
        'segment': np.random.choice(segments),
        'region': np.random.choice(regions),
        'join_date': join_date.strftime('%Y-%m-%d')
    })

# Generate sample product data
products_data = []
categories = ['Software', 'Hardware', 'Services', 'Consulting']
product_names = {
    'Software': ["CRM Suite", "Project Management Tool", "Cloud Storage Pro", "AI Analytics Software"],
    'Hardware': ["High-Performance Server", "Smart Router", "Workstation Pro", "Gaming Monitor"],
    'Services': ["IT Support", "Cloud Migration Service", "Data Backup", "System Maintenance"],
    'Consulting': ["Business Strategy Workshop", "IT Roadmap Planning", "Digital Transformation Advisory", "Process Optimization Consulting"]
}
base_prices = {'Software': 1000, 'Hardware': 500, 'Services': 200, 'Consulting': 2000}

for category, names in product_names.items():
    for name in names:
        products_data.append({
            'product_id': len(products_data) + 1,
            'product_name': name,
            'category': category,
            'unit_price': round(base_prices[category] * (0.8 + np.random.random() * 0.4), 2)
        })

# Generate sample sales data
sales_data = []
for i in range(3000):  # More sales to spread across years
    sale_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days + 1))
    customer_id = np.random.randint(1, len(company_names) + 1)
    product_id = np.random.randint(1, len(products_data) + 1)
    quantity = np.random.randint(1, 15)  # Higher quantity for realism
    
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
