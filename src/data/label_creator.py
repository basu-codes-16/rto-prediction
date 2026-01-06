import sqlite3
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import setup_logging, ensure_dir

logger = setup_logging(__name__)

class RTOLabelCreator:
    """Create RTO binary labels from order status."""
    
    def __init__(self, db_path: str = "data/rto_data.db"):
        self.conn = sqlite3.connect(db_path)
        logger.info(f"Connected to {db_path}")
    
    def create_rto_labels(self) -> pd.DataFrame:
        """Create binary RTO labels based on order_status."""
        
        # Define RTO mapping
        rto_statuses = ['return', 'rto', 'cancelled']
        delivered_statuses = ['delivered', 'shipped']
        
        query = """
        SELECT 
            o.*,
            f.order_status,
            f.pin,
            f.state as delivery_state,
            f.gst_amount,
            f.meesho_price,
            f.shipping_charges_total,
            f.price as final_price,
            f.order_date as forward_order_date,
            CASE 
                WHEN LOWER(f.order_status) IN ('return', 'rto', 'cancelled') THEN 1
                WHEN LOWER(f.order_status) IN ('delivered', 'shipped') THEN 0
                ELSE NULL
            END as is_rto
        FROM meesho_orders o
        INNER JOIN meesho_forward f 
            ON o.sub_order_no = f.sub_order_num
        WHERE f.order_status IS NOT NULL
        """
        
        df = pd.read_sql(query, self.conn)
        
        # Log label distribution
        logger.info(f"Total records: {len(df)}")
        logger.info(f"RTO (1): {df['is_rto'].sum()} ({df['is_rto'].mean()*100:.2f}%)")
        logger.info(f"Delivered (0): {(df['is_rto']==0).sum()} ({(df['is_rto']==0).mean()*100:.2f}%)")
        logger.info(f"Unknown (NULL): {df['is_rto'].isna().sum()}")
        
        # Remove nulls (Exchange status - ambiguous)
        df = df.dropna(subset=['is_rto'])
        logger.info(f"Final dataset: {len(df)} records after removing nulls")
        
        return df
    
    def save_labeled_data(self, output_path: str = "data/processed/meesho_labeled.csv"):
        """Create and save labeled dataset."""
        ensure_dir(Path(output_path).parent)
        
        df = self.create_rto_labels()
        df.to_csv(output_path, index=False)
        logger.info(f"Saved labeled data to {output_path}")
        
        return df
    
    def get_label_distribution(self) -> pd.DataFrame:
        """Analyze label distribution by various dimensions."""
        df = self.create_rto_labels()
        
        print("\n" + "="*60)
        print("RTO LABEL DISTRIBUTION")
        print("="*60)
        
        print("\n1. Overall Distribution:")
        print(df['is_rto'].value_counts())
        print(f"\nRTO Rate: {df['is_rto'].mean()*100:.2f}%")
        
        print("\n2. By State:")
        state_rto = df.groupby('delivery_state')['is_rto'].agg(['count', 'sum', 'mean'])
        state_rto.columns = ['Total', 'RTO_Count', 'RTO_Rate']
        state_rto['RTO_Rate'] = (state_rto['RTO_Rate'] * 100).round(2)
        print(state_rto.sort_values('RTO_Rate', ascending=False))
        
        print("\n3. By Product Category (top 10):")
        product_rto = df.groupby('product_name')['is_rto'].agg(['count', 'mean'])
        product_rto.columns = ['Total', 'RTO_Rate']
        product_rto['RTO_Rate'] = (product_rto['RTO_Rate'] * 100).round(2)
        print(product_rto.nlargest(10, 'RTO_Rate'))
        
        return df
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    creator = RTOLabelCreator()
    
    # Show distribution analysis
    df = creator.get_label_distribution()
    
    # Save labeled dataset
    creator.save_labeled_data()
    
    creator.close()
    
    print("\n✓ Labeled dataset ready for feature engineering!")