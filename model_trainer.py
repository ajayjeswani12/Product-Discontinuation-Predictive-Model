"""
Model training and evaluation module
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Any
import joblib
from pathlib import Path
from typing import Optional, Any

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, average_precision_score
)

# Optional imports for advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle model training, evaluation, and selection"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = []
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models to be evaluated"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',  # More stable than saga
                verbose=0  # Turn off verbose output
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=1  # Will be adjusted based on class imbalance
            )
        
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
        
        self.models = models
        return models
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for training - filter to pre-decision timepoints"""
        logger.info("Preparing training data...")
        
        # Filter to pre-decision data only (before final week)
        train_data = df[df['WeeksOut'] < -1].copy()

        # Keep the decision point (-1 week) for final evaluation
        eval_data = df[df['WeeksOut'] == -1].copy()
        
        logger.info(f"Training data: {len(train_data)} samples")
        logger.info(f"Evaluation data: {len(eval_data)} samples")
        logger.info(f"Training discontinuation rate: {train_data['DiscontinuedTF'].mean():.3f}")
        
        return train_data, eval_data
    
    def create_time_based_splits(self, df: pd.DataFrame, exclude_imbalanced: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create time-based train/validation/test split using CatEdition"""
        logger.info("Creating time-based splits...")
        
        # Check class distribution by edition
        edition_stats = df.groupby('CatEdition')['DiscontinuedTF'].agg(['count', 'mean']).round(3)
        logger.info(f"Class distribution by edition:\n{edition_stats}")
        
        df_work = df.copy()
        
        # Optionally filter out extremely imbalanced editions (>95% one class)
        if exclude_imbalanced:
            problematic = []
            for edition in edition_stats.index:
                rate = edition_stats.loc[edition, 'mean']
                if rate > 0.95 or rate < 0.05:
                    problematic.append(edition)
            
            if problematic:
                df_work = df_work[~df_work['CatEdition'].isin(problematic)]
                logger.info(f"Excluded imbalanced editions: {problematic}")
        
        # Get editions sorted chronologically
        editions = sorted(df_work['CatEdition'].unique())
        n_editions = len(editions)
        
        if n_editions < 3:
            raise ValueError(f"Need at least 3 editions for splits, got {n_editions}")
        
        # Split roughly 60% train, 20% val, 20% test
        n_train = max(1, int(n_editions * 0.6))
        n_val = max(1, int(n_editions * 0.2))
        
        train_editions = editions[:n_train]
        val_editions = editions[n_train:n_train + n_val]
        test_editions = editions[n_train + n_val:]
        
        train_df = df_work[df_work['CatEdition'].isin(train_editions)].copy()
        val_df = df_work[df_work['CatEdition'].isin(val_editions)].copy()
        test_df = df_work[df_work['CatEdition'].isin(test_editions)].copy()
        
        # Log results
        logger.info(f"Train: editions {train_editions} - {len(train_df)} samples")
        logger.info(f"Val: editions {val_editions} - {len(val_df)} samples") 
        logger.info(f"Test: editions {test_editions} - {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model and return metrics"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        return {
            'model_name': model_name,
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'avg_precision': average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
    
    def train_and_evaluate_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_val: pd.DataFrame, y_val: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train and evaluate all models"""
        logger.info("Training and evaluating models...")
        
        if not self.models:
            self.initialize_models()
        
        # Adjust for class imbalance
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        if 'xgboost' in self.models:
            self.models['xgboost'].set_params(scale_pos_weight=pos_weight)
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                val_metrics = self.evaluate_model(model, X_val, y_val, f"{model_name}_val")
                test_metrics = self.evaluate_model(model, X_test, y_test, f"{model_name}_test")
                
                results[model_name] = {
                    'val_roc_auc': val_metrics['roc_auc'],
                    'test_roc_auc': test_metrics['roc_auc'],
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
                
                logger.info(f"{model_name} - Val: {val_metrics['roc_auc']:.3f}, Test: {test_metrics['roc_auc']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        self.model_results = results
        return results
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from the specified model"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning(f"Cannot extract feature importance from {model_name}")
            return pd.DataFrame()
        
        if len(self.feature_columns) != len(importance):
            logger.warning("Feature columns length doesn't match importance length")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def select_best_model(self) -> Tuple[Any, str]:
        """Select best model based on validation ROC AUC"""
        best_score = 0
        best_name = None
        
        for model_name, metrics in self.model_results.items():
            val_auc = metrics['val_roc_auc']
            if val_auc and val_auc > best_score:
                best_score = val_auc
                best_name = model_name
        
        if best_name:
            self.best_model = self.models[best_name]
            self.best_model_name = best_name
            logger.info(f"Best model: {best_name} (Val AUC: {best_score:.3f})")
        
        return self.best_model, self.best_model_name
    
    def run_training_pipeline(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Complete training pipeline"""
        self.feature_columns = feature_columns
        
        # Prepare data
        train_data, eval_data = self.prepare_training_data(df)
        train_df, val_df, test_df = self.create_time_based_splits(train_data)
        
        # Extract features and targets
        X_train, y_train = train_df[feature_columns], train_df['DiscontinuedTF']
        X_val, y_val = val_df[feature_columns], val_df['DiscontinuedTF']
        X_test, y_test = test_df[feature_columns], test_df['DiscontinuedTF']
        
        # Train models
        results = self.train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Select best model
        self.select_best_model()
        
        return results
    
    def save_model(self, model_path: Path,feature_engineer: Optional[Any] = None):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No model trained")
        
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'feature_engineer': feature_engineer,
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model trained")
        
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        return predictions, probabilities