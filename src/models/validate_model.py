import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Load data
df = pd.read_csv('data/processed/meesho_features.csv')

# Load model artifacts
model = joblib.load('models/production/model.pkl')
scaler = joblib.load('models/production/scaler.pkl')
label_encoders = joblib.load('models/production/label_encoders.pkl')

# Encode categorical features (same as training)
for col, encoder in label_encoders.items():
    if col in df.columns:
        df[col + '_encoded'] = encoder.transform(df[col].astype(str))

# Define features (same as training)
feature_cols = [
    'day_of_week', 'day_of_month', 'month', 'is_weekend', 
    'is_month_start', 'is_month_end', 'is_metro', 'pin_rto_rate', 
    'state_rto_rate', 'pin_order_count', 'state_order_count',
    'quantity', 'meesho_price', 'final_price', 'shipping_charges_total',
    'price_per_unit', 'discount_amount', 'discount_pct', 
    'shipping_to_price_ratio', 'has_valid_pincode', 'has_state',
    'address_quality_score', 'product_rto_rate', 'product_order_count',
    'product_length', 'price_category_encoded', 'state_clean_encoded'
]

X = df[feature_cols].fillna(df[feature_cols].median())
y = df['is_rto']

# Scale
X_scaled = scaler.transform(X)

# 5-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS (5-Fold)")
print("="*70)

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=metric)
    print(f"\n{metric.upper()}:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std:  {scores.std():.4f}")
    print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print("✓ If mean CV scores ~ test scores (92%) → Model is reliable")
print("✗ If mean CV scores << test scores → Overfitting detected")