"""
Example usage of the Product Discontinuation Prediction system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import our modules
from main_pipeline import DiscontinuationPipeline
from predictor import DiscontinuationPredictor
from data_loader import DataLoader
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_full_training():
    """Example: Complete training pipeline"""
    print("=" * 60)
    print("EXAMPLE 1: COMPLETE TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = DiscontinuationPipeline()
    results = pipeline.run_full_pipeline(save_model=True)
    
    print(f"\nTraining Complete!")
    print(f"Best Model: {results['best_model']}")
    print(f"Best ROC AUC Score: {results['best_score']:.3f}")
    
    return results

def example_production_prediction():
    """Example: Using trained model for production predictions"""
    print("=" * 60)
    print("EXAMPLE 2: PRODUCTION PREDICTIONS")
    print("=" * 60)
    
    # Find the most recent trained model
    model_files = list(MODELS_DIR.glob("best_model_*.pkl"))
    if not model_files:
        print("No trained model found. Run training first!")
        return None
    
    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading model: {model_path}")
    
    # Load predictor
    predictor = DiscontinuationPredictor(model_path)
    
    # Load some sample data
    data_loader = DataLoader(PRODUCT_DETAILS_FILE, CATALOGUE_DISCONTINUATION_FILE)
    product_details, catalogue_data = data_loader.load_data()
    merged_data = data_loader.merge_datasets()
    
    # Get recent data (simulate current state)
    current_data = merged_data[
        (merged_data['WeeksOut'] > 0) & 
        (merged_data['WeeksOut'] <= 8)
    ].sample(n=min(50, len(merged_data)), random_state=42)
    
    print(f"Making predictions for {len(current_data)} products...")
    
    # Make predictions
    predictions = predictor.predict_discontinuation(current_data)
    
    print("\nTop 10 Highest Risk Products:")
    print(predictions[['ProductKey', 'discontinuation_probability', 'risk_category', 'confidence_level']].head(10).to_string(index=False))
    
    # Get business recommendations
    recommendations = predictor.get_business_recommendations(predictions)
    
    print(f"\nBusiness Recommendations:")
    print(f"  🔴 Immediate Action Required: {len(recommendations['immediate_action'])} products")
    print(f"