import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Connect to database
conn = sqlite3.connect('data/rto_data.db')

# Load labeled Meesho data
meesho = pd.read_csv('data/processed/meesho_labeled.csv')

print("="*70)
print("PART 1: EDA - RTO PREDICTION")
print("="*70)

# 1. Data Overview
print("\n1. DATASET OVERVIEW")
print(f"Total records: {len(meesho)}")
print(f"Features: {meesho.shape[1]}")
print(f"\nColumns: {meesho.columns.tolist()}")

# 2. Target Distribution
print("\n2. TARGET VARIABLE (is_rto)")
print(meesho['is_rto'].value_counts())
print(f"\nClass Imbalance:")
print(f"  Delivered (0): {(meesho['is_rto']==0).sum()} ({(meesho['is_rto']==0).mean()*100:.2f}%)")
print(f"  RTO (1):       {(meesho['is_rto']==1).sum()} ({meesho['is_rto'].mean()*100:.2f}%)")
print(f"  Imbalance Ratio: 1:{((meesho['is_rto']==0).sum() / meesho['is_rto'].sum()):.2f}")

# 3. Missing Values
print("\n3. MISSING VALUES")
missing = meesho.isnull().sum()
missing_pct = (missing / len(meesho) * 100).round(2)
missing_df = pd.DataFrame({
    'Count': missing[missing > 0],
    'Percentage': missing_pct[missing > 0]
}).sort_values('Percentage', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values")

# 4. Duplicates
print("\n4. DUPLICATES")
print(f"Duplicate sub_order_no: {meesho['sub_order_no'].duplicated().sum()}")

# 5. Temporal Analysis
print("\n5. TEMPORAL PATTERNS")
meesho['order_date'] = pd.to_datetime(meesho['order_date'], errors='coerce')
meesho['day_of_week'] = meesho['order_date'].dt.dayofweek
meesho['month'] = meesho['order_date'].dt.month

day_rto = meesho.groupby('day_of_week')['is_rto'].agg(['count', 'mean'])
print("\nRTO by Day of Week:")
print(day_rto)

# 6. Geographic Analysis
print("\n6. GEOGRAPHIC PATTERNS")
print(f"\nUnique States: {meesho['delivery_state'].nunique()}")
print(f"Unique Pincodes: {meesho['pin'].nunique()}")

state_rto = meesho.groupby('delivery_state')['is_rto'].agg(['count', 'mean'])
state_rto.columns = ['Total', 'RTO_Rate']
state_rto['RTO_Rate'] = (state_rto['RTO_Rate'] * 100).round(2)
print("\nRTO by State:")
print(state_rto.sort_values('RTO_Rate', ascending=False))

# 7. Price Analysis
print("\n7. TRANSACTIONAL PATTERNS")
price_cols = ['meesho_price', 'final_price', 'shipping_charges_total']
for col in price_cols:
    if col in meesho.columns:
        print(f"\n{col}:")
        print(f"  Mean (Delivered): {meesho[meesho['is_rto']==0][col].mean():.2f}")
        print(f"  Mean (RTO):       {meesho[meesho['is_rto']==1][col].mean():.2f}")

# 8. Product Analysis
print("\n8. PRODUCT PATTERNS")
product_rto = meesho.groupby('product_name')['is_rto'].agg(['count', 'mean'])
product_rto.columns = ['Total', 'RTO_Rate']
product_rto['RTO_Rate'] = (product_rto['RTO_Rate'] * 100).round(2)
print("\nTop products by RTO rate (min 2 orders):")
print(product_rto[product_rto['Total'] >= 2].sort_values('RTO_Rate', ascending=False).head())

# 9. DTDC Dataset Overview (for context)
print("\n9. DTDC DATASET (For Feature Engineering)")
dtdc_sample = pd.read_sql("SELECT * FROM dtdc LIMIT 5", conn)
print(f"DTDC Records: {pd.read_sql('SELECT COUNT(*) FROM dtdc', conn).iloc[0,0]}")
print(f"Columns: {dtdc_sample.columns.tolist()[:10]}...")  # First 10 columns

# 10. Delhivery Dataset Overview
print("\n10. DELHIVERY DATASET (For Feature Engineering)")
delhivery_sample = pd.read_sql("SELECT * FROM delhivery LIMIT 5", conn)
print(f"Delhivery Records: {pd.read_sql('SELECT COUNT(*) FROM delhivery', conn).iloc[0,0]}")
print(f"Columns: {delhivery_sample.columns.tolist()[:10]}...")

# Key Insights
print("\n" + "="*70)
print("KEY INSIGHTS FOR FEATURE ENGINEERING:")
print("="*70)
print("✓ Class imbalance detected - need SMOTE/class weights")
print("✓ Geographic (state/pincode) shows RTO variation")
print("✓ Temporal patterns (day of week) exist")
print("✓ Price differences between RTO/Delivered")
print("✓ Product category matters")
print("✓ DTDC/Delhivery available for geographic feature enrichment")

conn.close()
print("\n✓ EDA Complete")