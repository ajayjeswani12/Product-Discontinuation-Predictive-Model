"""
Main pipeline for product discontinuation prediction
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
import json
from typing import Dict, Any

# Import our modules
from config import *
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from predictor import DiscontinuationPredictor

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DiscontinuationPipeline:
    """Complete pipeline for product discontinuation prediction"""
    
    def __init__(self):
        self.data_loader = DataLoader(PRODUCT_DETAILS_FILE, CATALOGUE_DISCONTINUATION_FILE)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(random_state=RANDOM_STATE)
        self.predictor = DiscontinuationPredictor()
        
        self.raw_data = None
        self.processed_data = None
        self.model_results = None
    
    def run_full_pipeline(self, save_model: bool = True) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        logger.info("Starting  Product Discontinuation Prediction Pipeline")
        
        # Step 1: Load and validate data
        logger.info("=" * 50)
        logger.info("STEP 1: DATA LOADING AND VALIDATION")
        logger.info("=" * 50)
        
        product_details, catalogue_data = self.data_loader.load_data()
        merged_data = self.data_loader.merge_datasets()
        data_summary = self.data_loader.get_data_summary()

        print("=== DATA DIAGNOSTICS ===")
        print(f"Dataset shape: {merged_data.shape}")
        print(f"Memory usage: {merged_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Data types:\n{merged_data.dtypes.value_counts()}")
        
        logger.info("Data Summary:")
        for key, value in data_summary.items():
            logger.info(f"  {key}: {value}")
        
        self.raw_data = merged_data
        
        # Step 2: Feature Engineering
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        # Create features
        df_features = self.feature_engineer.create_features(merged_data)
        df_features = self.feature_engineer.create_aggregated_features(df_features)
        
        # Prepare training and evaluation data
        train_data, eval_data = self.model_trainer.prepare_training_data(df_features)
        
        self.processed_data = {
            'full_features': df_features,
            'train_data': train_data,
            'eval_data': eval_data
        }
        
        # Step 3: Model Training and Selection
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL TRAINING AND EVALUATION")
        logger.info("=" * 50)
        
        # Create time-based train/test split
        train_df, val_df, test_df = self.model_trainer.create_time_based_splits(train_data)
        
        # Prepare features for modeling
        X_train, feature_cols = self.feature_engineer.prepare_model_features(train_df, training=True)
        X_val, _ = self.feature_engineer.prepare_model_features(val_df, training=False)
        X_test, _ = self.feature_engineer.prepare_model_features(test_df, training=False)
        
        y_train = train_df['DiscontinuedTF']
        y_val = val_df['DiscontinuedTF'] 
        y_test = test_df['DiscontinuedTF']
        
        # Store feature columns in trainer
        self.model_trainer.feature_columns = feature_cols
        
        # Train and evaluate models
        results = self.model_trainer.train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)
        best_model, best_model_name = self.model_trainer.select_best_model()
        
        self.model_results = results
        
        # Step 4: Model Analysis
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL ANALYSIS")
        logger.info("=" * 50)
        
        # Feature importance
        try:
            feature_importance = self.model_trainer.get_feature_importance()
            logger.info("Top 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        # Model performance summary
        logger.info("\nModel Performance Summary:")
        for model_name, metrics in results.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Test ROC AUC: {metrics['test_roc_auc']:.3f}")
            logger.info(f"    Val ROC AUC: {metrics['val_roc_auc']:.3f}")
            
            
            # Classification report summary
            report = metrics['test_metrics']['classification_report']  # or 'val_metrics'
            logger.info(f"    Precision (Discontinued): {report['True']['precision']:.3f}")
            logger.info(f"    Recall (Discontinued): {report['True']['recall']:.3f}")
        
        # Step 5: Save Model
        if save_model:
            logger.info("=" * 50)
            logger.info("STEP 5: SAVING MODEL")
            logger.info("=" * 50)
            
            model_path = MODELS_DIR / f"best_model_{best_model_name}.pkl"
            self.model_trainer.save_model(model_path,self.feature_engineer)
            
            # Save feature importance
            try:
                importance_path = RESULTS_DIR / "feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                logger.info(f"Feature importance saved to {importance_path}")
            except:
                pass
            
            # Save results summary
            results_path = RESULTS_DIR / "model_results.json"
            with open(results_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {}
                for model_name, metrics in results.items():
                    json_results[model_name] = {
                        'test_roc_auc': float(metrics['test_roc_auc']) if metrics['test_roc_auc'] else None,
                     
                    }
                
                json.dump({
                    'best_model': best_model_name,
                    'model_results': json_results,
                    'data_summary': {k: v for k, v in data_summary.items() if isinstance(v, (int, float, str, bool))}
                }, f, indent=2)
            
            logger.info(f"Results summary saved to {results_path}")
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best ROC AUC: {results[best_model_name]['test_roc_auc']:.3f}")
        
        return {
            'best_model': best_model_name,
            'best_score': results[best_model_name]['test_roc_auc'],
            'model_results': results,
            'data_summary': data_summary
        }
    
    def run_prediction_example(self, model_path: Path = None):
        """Run example predictions using the trained model"""
        logger.info("Running prediction example...")
        
        if model_path is None:
            # Find the most recent model
            model_files = list(MODELS_DIR.glob("best_model_*.pkl"))
            if not model_files:
                raise FileNotFoundError("No trained model found. Run training first.")
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Load model
        self.predictor.load_model(model_path)
        
        # Use recent raw data (positive WeeksOut for realistic prediction scenario)
        if self.raw_data is None:
            logger.info("Loading raw data for prediction example...")
            product_details, catalogue_data = self.data_loader.load_data()
            merged_data = self.data_loader.merge_datasets()
            self.raw_data = merged_data
        
        recent_raw = self.raw_data[
            (self.raw_data['WeeksOut'] < 0) &
            (self.raw_data['WeeksOut'] >= -8)
        ].sample(n=min(20, len(self.raw_data)), random_state=RANDOM_STATE)
        
        # Pass raw data to predictor
        predictions = self.predictor.predict_discontinuation(recent_raw)
        
        logger.info("Sample Predictions:")
        logger.info(predictions[['ProductKey', 'discontinuation_probability', 'risk_category', 'confidence_level']].head(10).to_string(index=False))
        
        # Generate business recommendations
        recommendations = self.predictor.get_business_recommendations(predictions)
        
        logger.info(f"\nBusiness Recommendations Summary:")
        logger.info(f"  Immediate Action Required: {len(recommendations['immediate_action'])} products")
        logger.info(f"  Monitor Closely: {len(recommendations['monitor_closely'])} products")
        logger.info(f"  Standard Monitoring: {len(recommendations['standard_monitoring'])} products")
        
        return predictions, recommendations

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description=" Product Discontinuation Prediction Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both",
                       help="Mode to run: train model, make predictions, or both")
    parser.add_argument("--model-path", type=Path, help="Path to saved model (for prediction mode)")
    parser.add_argument("--no-save", action="store_true", help="Don't save the trained model")
    
    args = parser.parse_args()
    
    pipeline = DiscontinuationPipeline()
    
    try:
        if args.mode in ["train", "both"]:
            logger.info("Running training pipeline...")
            results = pipeline.run_full_pipeline(save_model=not args.no_save)
        
        if args.mode in ["predict", "both"]:
            logger.info("Running prediction example...")
            predictions, recommendations = pipeline.run_prediction_example(args.model_path)
            
            # Save predictions example
            pred_path = RESULTS_DIR / "sample_predictions.csv"
            predictions.to_csv(pred_path, index=False)
            logger.info(f"Sample predictions saved to {pred_path}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()