import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import setup_logging, ensure_dir

logger = setup_logging(__name__)

class FeatureEngineer:
    """Build features for RTO prediction."""
    
    def __init__(self):
        self.metro_pincodes = ['110', '400', '560', '600', '700', '500']  # Delhi, Mumbai, Bangalore, Chennai, Kolkata, Hyderabad
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from order_date."""
        logger.info("Creating temporal features")
        
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        
        df['day_of_week'] = df['order_date'].dt.dayofweek
        df['day_of_month'] = df['order_date'].dt.day
        df['month'] = df['order_date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        return df
    
    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic features from pincode and state."""
        logger.info("Creating geographic features")
        
        # Clean pincode
        df['pin'] = df['pin'].astype(str).str.zfill(6)
        df['pin_prefix'] = df['pin'].str[:3]
        
        # Metro city indicator
        df['is_metro'] = df['pin_prefix'].isin(self.metro_pincodes).astype(int)
        
        # State encoding (will be label encoded later)
        df['state_clean'] = df['delivery_state'].str.strip().str.title()
        
        # Pincode-level aggregations (historical RTO rate)
        pin_rto = df.groupby('pin')['is_rto'].agg(['mean', 'count'])
        pin_rto.columns = ['pin_rto_rate', 'pin_order_count']
        df = df.merge(pin_rto, left_on='pin', right_index=True, how='left')
        
        # State-level aggregations
        state_rto = df.groupby('state_clean')['is_rto'].agg(['mean', 'count'])
        state_rto.columns = ['state_rto_rate', 'state_order_count']
        df = df.merge(state_rto, left_on='state_clean', right_index=True, how='left')
        
        return df
    
    def create_transactional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from transaction data."""
        logger.info("Creating transactional features")
        
        # Price features
        df['price_per_unit'] = df['final_price'] / df['quantity'].replace(0, 1)
        df['discount_amount'] = df['meesho_price'] - df['final_price']
        df['discount_pct'] = (df['discount_amount'] / df['meesho_price'].replace(0, 1) * 100).clip(0, 100)
        
        # Shipping ratio
        df['shipping_to_price_ratio'] = df['shipping_charges_total'] / df['final_price'].replace(0, 1)
        
        # Price bins
        df['price_category'] = pd.cut(df['final_price'], 
                                       bins=[0, 200, 500, 1000, float('inf')],
                                       labels=['low', 'medium', 'high', 'premium'])
        
        return df
    
    def create_address_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create address quality score using simple NLP."""
        logger.info("Creating address quality features")
        
        # For Meesho, we don't have full address, but we have state and pin
        # Address completeness score based on available data
        df['has_valid_pincode'] = df['pin'].str.match(r'^\d{6}$', na=False).astype(int)
        df['has_state'] = df['delivery_state'].notna().astype(int)
        
        # Address quality score (0-1)
        df['address_quality_score'] = (
            df['has_valid_pincode'] * 0.6 +
            df['has_state'] * 0.4
        )
        
        return df
    
    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product-based features."""
        logger.info("Creating product features")
        
        # Product RTO rate
        product_rto = df.groupby('product_name')['is_rto'].agg(['mean', 'count'])
        product_rto.columns = ['product_rto_rate', 'product_order_count']
        df = df.merge(product_rto, left_on='product_name', right_index=True, how='left')
        
        # Product category (extract from name - simple approach)
        df['product_length'] = df['product_name'].str.len()
        
        return df
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features."""
        logger.info("Starting feature engineering")
        
        df = self.create_temporal_features(df)
        df = self.create_geographic_features(df)
        df = self.create_transactional_features(df)
        df = self.create_address_quality_features(df)
        df = self.create_product_features(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df

def main():
    """Main feature engineering pipeline."""
    
    # Load labeled data
    df = pd.read_csv('data/processed/meesho_labeled.csv')
    logger.info(f"Loaded data: {df.shape}")
    
    # Build features
    fe = FeatureEngineer()
    df_features = fe.build_all_features(df)
    
    # Save
    ensure_dir('data/processed')
    output_path = 'data/processed/meesho_features.csv'
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*70)
    print(f"Input shape: {df.shape}")
    print(f"Output shape: {df_features.shape}")
    print(f"New features created: {df_features.shape[1] - df.shape[1]}")
    print(f"\nNew feature columns:")
    new_cols = set(df_features.columns) - set(df.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    return df_features

if __name__ == "__main__":
    main()