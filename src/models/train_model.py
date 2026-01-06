import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import json
sys.path.append(str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Optional imports for XGBoost/LightGBM
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import mlflow
import mlflow.sklearn

from src.utils import setup_logging, load_config, ensure_dir

logger = setup_logging(__name__)

class RTOModelTrainer:
    """Train and evaluate RTO prediction models."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.models = {}
        self.results = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # MLflow setup
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for training."""
        logger.info("Preparing data for training")
        
        # Select features
        feature_cols = [
            # Temporal
            'day_of_week', 'day_of_month', 'month', 'is_weekend', 
            'is_month_start', 'is_month_end',
            # Geographic
            'is_metro', 'pin_rto_rate', 'state_rto_rate',
            'pin_order_count', 'state_order_count',
            # Transactional
            'quantity', 'meesho_price', 'final_price', 
            'shipping_charges_total', 'price_per_unit',
            'discount_amount', 'discount_pct', 'shipping_to_price_ratio',
            # Address
            'has_valid_pincode', 'has_state', 'address_quality_score',
            # Product
            'product_rto_rate', 'product_order_count', 'product_length'
        ]
        
        # Categorical features to encode
        cat_features = ['price_category', 'state_clean']
        
        # Encode categorical
        for col in cat_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                feature_cols.append(col + '_encoded')
        
        # Handle missing values
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df['is_rto']
        
        logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y):
        """Split data into train/test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train RTO rate: {y_train.mean():.2%}")
        logger.info(f"Test RTO rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for class balancing."""
        logger.info("Applying SMOTE")
        
        # Check if we have enough samples for SMOTE
        minority_class_count = min(y_train.value_counts())
        majority_class_count = max(y_train.value_counts())
        
        if minority_class_count < 6:
            logger.warning("Too few samples for SMOTE - skipping")
            return X_train, y_train
        
        # Use auto strategy for small datasets
        try:
            smote = SMOTE(
                sampling_strategy='auto',  # Changed from 0.5 to auto
                random_state=self.config['model']['random_state'],
                k_neighbors=min(5, minority_class_count - 1)  # Adjust k_neighbors
            )
            
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            
            logger.info(f"Before SMOTE: {len(y_train)}, After: {len(y_train_sm)}")
            logger.info(f"Class distribution after SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")
            
            return X_train_sm, y_train_sm
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Using original data with class weights.")
            return X_train, y_train
    
    def scale_features(self, X_train, X_test):
        """Scale features."""
        logger.info("Scaling features")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models."""
        logger.info("Training models")
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=1.3,
                random_state=42
            )
        else:
            logger.warning("XGBoost not available - skipping")
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                is_unbalance=True,
                random_state=42,
                verbose=-1
            )
        else:
            logger.warning("LightGBM not available - skipping")
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            with mlflow.start_run(run_name=name):
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Evaluate
                results = self.evaluate_model(y_test, y_pred, y_pred_proba, name)
                
                # Separate metrics and non-metrics
                metrics_to_log = {k: v for k, v in results.items() if k != 'confusion_matrix'}
                
                # Log to MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics_to_log)
                mlflow.sklearn.log_model(model, name)
                
                # Log confusion matrix separately as artifact
                cm_dict = {'confusion_matrix': results['confusion_matrix']}
                mlflow.log_dict(cm_dict, f"{name}_confusion_matrix.json")
                
                # Store
                self.models[name] = model
                self.results[name] = results
                
                logger.info(f"{name} - F1: {results['f1_score']:.4f}, AUC: {results['roc_auc']:.4f}")
    
    def evaluate_model(self, y_test, y_pred, y_pred_proba, model_name):
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # PR AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        results['pr_auc'] = auc(recall_vals, precision_vals)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        return results
    
    def save_best_model(self):
        """Save the best performing model."""
        # Select best by F1 score
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name}")
        
        # Save model artifacts
        model_dir = ensure_dir('models/production')
        joblib.dump(best_model, model_dir / 'model.pkl')
        joblib.dump(self.scaler, model_dir / 'scaler.pkl')
        joblib.dump(self.label_encoders, model_dir / 'label_encoders.pkl')
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'model_params': best_model.get_params(),
            'performance': self.results[best_model_name],
            'feature_importance': self.get_feature_importance(best_model)
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved best model to {model_dir}")
        
        return best_model_name, best_model
    
    def get_feature_importance(self, model):
        """Get feature importance if available."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            return model.coef_[0].tolist()
        return None
    
    def print_results_summary(self):
        """Print training results summary."""
        print("\n" + "="*70)
        print("MODEL TRAINING RESULTS")
        print("="*70)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']]
        results_df = results_df.round(4)
        
        print("\nModel Performance:")
        print(results_df.to_string())
        
        best_model = max(self.results, key=lambda x: self.results[x]['f1_score'])
        print(f"\n✓ Best Model: {best_model} (F1: {self.results[best_model]['f1_score']:.4f})")

def main():
    """Main training pipeline."""
    
    # Load data
    df = pd.read_csv('data/processed/meesho_features.csv')
    logger.info(f"Loaded features: {df.shape}")
    
    # Initialize trainer
    trainer = RTOModelTrainer()
    
    # Prepare data
    X, y, feature_cols = trainer.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Apply SMOTE
    X_train_sm, y_train_sm = trainer.apply_smote(X_train, y_train)
    
    # Scale features
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train_sm, X_test)
    
    # Train models
    trainer.train_models(X_train_scaled, y_train_sm, X_test_scaled, y_test)
    
    # Print results
    trainer.print_results_summary()
    
    # Save best model
    best_name, best_model = trainer.save_best_model()
    
    print("\n✓ Training complete! Check mlruns/ for experiment tracking")
    print(f"✓ Best model saved to models/production/")

if __name__ == "__main__":
    main()