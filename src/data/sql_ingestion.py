import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import setup_logging, load_config

logger = setup_logging(__name__)

class SQLDataIngestion:
    """Ingest CSV data into SQLite database."""
    
    def __init__(self, db_path: str = "data/rto_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
    
    def ingest_csv(self, csv_path: str, table_name: str, 
                   chunksize: Optional[int] = 10000) -> None:
        """Ingest CSV into SQLite table."""
        try:
            logger.info(f"Ingesting {csv_path} → {table_name}")
            
            # Read and ingest in chunks
            chunks_ingested = 0
            for chunk in pd.read_csv(csv_path, chunksize=chunksize):
                # Clean column names
                chunk.columns = chunk.columns.str.strip().str.lower().str.replace(' ', '_')
                
                # Append to table
                chunk.to_sql(table_name, self.conn, if_exists='append', index=False)
                chunks_ingested += 1
                
                if chunks_ingested % 10 == 0:
                    logger.info(f"  Processed {chunks_ingested * chunksize} rows")
            
            # Get final count
            count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", self.conn)
            logger.info(f"✓ Ingested {count['cnt'][0]} rows into {table_name}")
            
        except Exception as e:
            logger.error(f"Error ingesting {csv_path}: {e}")
            raise
    
    def create_indexes(self) -> None:
        """Create indexes for faster joins."""
        cursor = self.conn.cursor()
        
        indexes = [
            # Meesho tables
            "CREATE INDEX IF NOT EXISTS idx_orders_suborder ON meesho_orders(sub_order_no)",
            "CREATE INDEX IF NOT EXISTS idx_forward_suborder ON meesho_forward(sub_order_num)",
            "CREATE INDEX IF NOT EXISTS idx_forward_status ON meesho_forward(order_status)",
            
            # DTDC
            "CREATE INDEX IF NOT EXISTS idx_dtdc_consignment ON dtdc(consignment_no)",
            "CREATE INDEX IF NOT EXISTS idx_dtdc_sender_pin ON dtdc(sender_pincode)",
            "CREATE INDEX IF NOT EXISTS idx_dtdc_receiver_pin ON dtdc(receiver_pincode)",
            
            # Delhivery (if exists)
            "CREATE INDEX IF NOT EXISTS idx_delhivery_date ON delhivery(date)",
        ]
        
        for idx_query in indexes:
            try:
                cursor.execute(idx_query)
                logger.info(f"✓ {idx_query.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")
        
        self.conn.commit()
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get table schema and row count."""
        schema = pd.read_sql(f"PRAGMA table_info({table_name})", self.conn)
        count = pd.read_sql(f"SELECT COUNT(*) as row_count FROM {table_name}", self.conn)
        
        print(f"\n{'='*60}")
        print(f"TABLE: {table_name}")
        print(f"{'='*60}")
        print(f"Rows: {count['row_count'][0]:,}")
        print(f"\nColumns:\n{schema[['name', 'type']].to_string(index=False)}")
        
        return schema
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return pd.read_sql(sql, self.conn)
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")

def ingest_all_data():
    """Main ingestion pipeline."""
    ingestion = SQLDataIngestion()
    
    raw_path = Path("data/raw")
    
    # Ingest datasets - exact filenames from screenshot
    datasets = [
        (raw_path / "meesho Orders Aug.csv", "meesho_orders"),
        (raw_path / "meesho ForwardReports.csv", "meesho_forward"),
        (raw_path / "Dataset_Generator_for_DTDC.csv", "dtdc"),
        (raw_path / "delhivery_data.csv", "delhivery"),  # Large file - chunked
    ]
    
    for csv_path, table_name in datasets:
        if csv_path.exists():
            ingestion.ingest_csv(str(csv_path), table_name)
        else:
            logger.warning(f"File not found: {csv_path}")
    
    # Create indexes for performance
    ingestion.create_indexes()
    
    # Show table info
    for _, table_name in datasets:
        try:
            ingestion.get_table_info(table_name)
        except:
            pass
    
    return ingestion

if __name__ == "__main__":
    db = ingest_all_data()
    
    print("\n" + "="*60)
    print("DATABASE READY FOR QUERIES!")
    print("="*60)
    
    # Example query
    print("\nSample query - Order status distribution:")
    result = db.query("""
        SELECT order_status, COUNT(*) as count 
        FROM meesho_forward 
        GROUP BY order_status
    """)
    print(result)
    
    db.close()