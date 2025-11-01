"""
Prediction-only pipeline for Argos product discontinuation
This script loads a trained model and predicts on products with catedition = 94
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
import json
from typing import Dict, Any, Tuple

# Import our modules
from config import *
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from predictor import DiscontinuationPredictor

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArgosPredictionOnlyPipeline:
    """Pipeline for making predictions only - no training"""
    
    def __init__(self):
        self.data_loader = DataLoader(PRODUCT_DETAILS_FILE, CATALOGUE_DISCONTINUATION_FILE)
        self.predictor = DiscontinuationPredictor()
        self.raw_data = None
    
    def load_raw_data(self):
        """Load and merge raw data"""
        logger.info("Loading raw data...")
        
        product_details, catalogue_data = self.data_loader.load_data()
        merged_data = self.data_loader.merge_datasets()
        
        self.raw_data = merged_data
        logger.info(f"Loaded data with shape: {merged_data.shape}")
        
        return merged_data
    
    def predict_products(self, model_path: Path = None, CatEdition: int = 94, sample_size: int = 20):
        """
        Predict discontinuation for products with specified catedition value
        
        Args:
            model_path: Path to the trained model file
            catedition: Filter data by this catedition value (default: 94)
            sample_size: Number of products to predict (default: 20)
        """
        logger.info("=" * 60)
        logger.info("ARGOS PRODUCT DISCONTINUATION PREDICTION")
        logger.info("=" * 60)
        
        # Step 1: Load trained model
        logger.info("Step 1: Loading trained model...")
        
        if model_path is None:
            # Find the most recent model
            model_files = list(MODELS_DIR.glob("best_model_*.pkl"))
            if not model_files:
                raise FileNotFoundError("No trained model found. Please specify model path or train a model first.")
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Using most recent model: {model_path.name}")
        else:
            logger.info(f"Using specified model: {model_path}")
        
        self.predictor.load_model(model_path)
        logger.info("✓ Model loaded successfully")
        
        # Step 2: Load raw data
        logger.info("\nStep 2: Loading raw data...")
        if self.raw_data is None:
            self.load_raw_data()
        
        logger.info(f"✓ Raw data loaded: {len(self.raw_data)} total records")
        
        # Step 3: Filter data
        logger.info(f"\nStep 3: Filtering data for catedition = {CatEdition}")
        
        # Filter for specified catedition
        target_data = self.raw_data[self.raw_data['CatEdition'] == CatEdition]
        
        if len(target_data) == 0:
            logger.warning(f"No data found with catedition = {CatEdition}. Please check the column name and values.")
            logger.info("Available catedition values:")
            logger.info(self.raw_data['catedition'].value_counts().head(10))
            raise ValueError(f"No data found with catedition = {CatEdition}")
        
        logger.info(f"Found {len(target_data)} products with catedition = {CatEdition}")
        
        # Sample the data
        if len(target_data) > sample_size:
            sample_data = target_data.sample(n=sample_size, random_state=RANDOM_STATE)
            logger.info(f"✓ Randomly sampled {sample_size} products for prediction")
        else:
            sample_data = target_data
            logger.info(f"✓ Using all {len(sample_data)} available products")
        
        # Step 4: Generate predictions
        logger.info(f"\nStep 4: Generating predictions...")
        
        predictions = self.predictor.predict_discontinuation(sample_data)
        logger.info(f"✓ Generated predictions for {len(predictions)} products")
        
        # Step 5: Generate business recommendations
        logger.info(f"\nStep 5: Generating business recommendations...")
        recommendations = self.predictor.get_business_recommendations(predictions)
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION RESULTS")
        logger.info("=" * 60)
        
        # Show sample predictions
        logger.info("\nTop 10 Products by Discontinuation Risk:")
        display_cols = ['ProductKey', 'discontinuation_probability', 'risk_category', 'confidence_level']
        if all(col in predictions.columns for col in display_cols):
            top_predictions = predictions.nlargest(10, 'discontinuation_probability')
            logger.info("\n" + top_predictions[display_cols].to_string(index=False))
        
        # Show risk distribution
        logger.info(f"\nRisk Category Distribution:")
        risk_dist = predictions['risk_category'].value_counts()
        for category, count in risk_dist.items():
            percentage = (count / len(predictions)) * 100
            logger.info(f"  {category}: {count} products ({percentage:.1f}%)")
        
        # Show business recommendations summary
        logger.info(f"\nBusiness Recommendations Summary:")
        logger.info(f"  🔴 Immediate Action Required: {len(recommendations['immediate_action'])} products")
        logger.info(f"  🟡 Monitor Closely: {len(recommendations['monitor_closely'])} products")
        logger.info(f"  🟢 Standard Monitoring: {len(recommendations['standard_monitoring'])} products")
        
        # Show high-risk products if any
        high_risk = predictions[predictions['risk_category'] == 'HIGH']
        if len(high_risk) > 0:
            logger.info(f"\n🚨 HIGH RISK PRODUCTS requiring immediate attention:")
            for _, row in high_risk.head(5).iterrows():
                prob = row['discontinuation_probability']
                logger.info(f"  - ProductKey {row['ProductKey']}: {prob:.1%} probability")
        
        # Step 6: Save results
        logger.info(f"\nStep 6: Saving results...")
        
        # Save detailed predictions
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        pred_filename = f"predictions_catedition_{CatEdition}_{timestamp}.csv"
        pred_path = RESULTS_DIR / pred_filename
        
        predictions.to_csv(pred_path, index=False)
        logger.info(f"✓ Detailed predictions saved to: {pred_path}")
        
        # Save recommendations summary
        rec_filename = f"recommendations_catedition_{CatEdition}_{timestamp}.json"
        rec_path = RESULTS_DIR / rec_filename
        
        # Convert recommendations to serializable format
        serializable_recs = {}
        for key, products in recommendations.items():
            if hasattr(products, 'to_dict'):
                serializable_recs[key] = products.to_dict('records')
            else:
                serializable_recs[key] = list(products)
        
        with open(rec_path, 'w') as f:
            json.dump({
                'CatEdition': CatEdition,
                'total_products_analyzed': len(predictions),
                'prediction_timestamp': timestamp,
                'recommendations': serializable_recs,
                'risk_distribution': risk_dist.to_dict()
            }, f, indent=2, default=str)
        
        logger.info(f"✓ Recommendations saved to: {rec_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return predictions, recommendations

def main():
    """Main function to run prediction pipeline"""
    parser = argparse.ArgumentParser(
        description="Argos Product Discontinuation Prediction Pipeline - Prediction Only",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-path", 
        type=Path, 
        help="Path to saved model file. If not specified, uses most recent model."
    )
    parser.add_argument(
        "--CatEdition", 
        type=int, 
        default=94, 
        help="Filter data by catedition value"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=20, 
        help="Maximum number of products to predict"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create pipeline
    pipeline = ArgosPredictionOnlyPipeline()
    
    try:
        # Run predictions
        predictions, recommendations = pipeline.predict_products(
            model_path=args.model_path, 
            CatEdition=args.CatEdition, 
            sample_size=args.sample_size
        )
        
        print(f"\n✅ SUCCESS: Predicted discontinuation risk for {len(predictions)} products")
        print(f"📊 Results saved to: {RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"❌ Prediction pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()