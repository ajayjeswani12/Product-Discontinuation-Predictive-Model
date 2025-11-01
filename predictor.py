"""
Production predictor interface for product discontinuation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import joblib

from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class DiscontinuationPredictor:
    """Production-ready predictor for product discontinuation"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = []
        self.model_name = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: Path):
        """Load trained model and feature engineering pipeline"""
        logger.info(f"Loading model from {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_columns = model_data['feature_columns']
        self.feature_engineer = model_data['feature_engineer']
        
        logger.info(f"Loaded {self.model_name} model with {len(self.feature_columns)} features")
    
    def predict_discontinuation(self, product_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict discontinuation probability for products
        
        Args:
            product_data: DataFrame with columns matching training data structure
            
        Returns:
            DataFrame with ProductKey, discontinuation probability, and risk category
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        logger.info(f"Predicting discontinuation for {len(product_data)} products...")
        
        # Feature engineering
        df_features = self.feature_engineer.create_features(product_data)
        df_features = self.feature_engineer.create_aggregated_features(df_features)
        X, _ = self.feature_engineer.prepare_model_features(df_features, training=False)
        
        # Ensure we have the same features as training
        missing_features = set(self.feature_columns) - set(X.columns)
        extra_features = set(X.columns) - set(self.feature_columns)
        
        if missing_features:
            logger.warning(f"Missing features (will be filled with 0): {missing_features}")
            for feature in missing_features:
                X[feature] = 0
        
        if extra_features:
            logger.warning(f"Extra features (will be dropped): {extra_features}")
            X = X.drop(columns=list(extra_features))
        
        # Reorder columns to match training
        X = X[self.feature_columns]
        
        # Make predictions
        try:
            probabilities = self.model.predict_proba(X)[:, 1]
            predictions = self.model.predict(X)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
        
        # Create results dataframe
        results = pd.DataFrame({
            'ProductKey': product_data['ProductKey'],
            'discontinuation_probability': probabilities,
            'predicted_discontinued': predictions,
            'risk_category': self._categorize_risk(probabilities),
            'confidence_level': self._calculate_confidence(probabilities)
        })
        
        # Add business context
        if 'WeeksOut' in product_data.columns:
            results['weeks_to_decision'] = product_data['WeeksOut']
        
        if 'ActualsPerWeek' in product_data.columns:
            results['current_sales_volume'] = product_data['ActualsPerWeek']
        
        if 'SalePriceIncVAT' in product_data.columns:
            results['price'] = product_data['SalePriceIncVAT']
        
        # Sort by discontinuation probability (highest risk first)
        results = results.sort_values('discontinuation_probability', ascending=False)
        
        logger.info(f"Predictions complete. High risk products: {(results['risk_category'] == 'High Risk').sum()}")
        return results
    
    def _categorize_risk(self, probabilities: np.ndarray) -> List[str]:
        """Categorize products by discontinuation risk"""
        categories = []
        for prob in probabilities:
            if prob >= 0.8:
                categories.append('High Risk')
            elif prob >= 0.6:
                categories.append('Medium Risk')
            elif prob >= 0.4:
                categories.append('Low-Medium Risk')
            else:
                categories.append('Low Risk')
        return categories
    
    def _calculate_confidence(self, probabilities: np.ndarray) -> List[str]:
        """Calculate prediction confidence based on probability distance from 0.5"""
        confidence = []
        for prob in probabilities:
            distance = abs(prob - 0.5)
            if distance >= 0.4:
                confidence.append('High')
            elif distance >= 0.2:
                confidence.append('Medium')
            else:
                confidence.append('Low')
        return confidence
    
    def get_business_recommendations(self, predictions: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Generate business recommendations based on predictions"""
        recommendations = {
            'immediate_action': [],
            'monitor_closely': [],
            'standard_monitoring': []
        }
        
        for _, row in predictions.iterrows():
            product_info = {
                'ProductKey': row['ProductKey'],
                'discontinuation_probability': row['discontinuation_probability'],
                'risk_category': row['risk_category']
            }
            
            # Add business context if available
            if 'current_sales_volume' in row:
                product_info['sales_volume'] = row['current_sales_volume']
            if 'weeks_to_decision' in row:
                product_info['weeks_to_decision'] = row['weeks_to_decision']
            
            # Categorize recommendations
            if row['discontinuation_probability'] >= 0.8 and row['confidence_level'] == 'High':
                recommendations['immediate_action'].append({
                    **product_info,
                    'action': 'Stop restocking - high discontinuation probability',
                    'priority': 'Critical'
                })
            elif row['discontinuation_probability'] >= 0.6:
                recommendations['monitor_closely'].append({
                    **product_info,
                    'action': 'Reduce stock levels and monitor daily',
                    'priority': 'High'
                })
            elif row['discontinuation_probability'] >= 0.4:
                recommendations['standard_monitoring'].append({
                    **product_info,
                    'action': 'Continue standard restocking with weekly review',
                    'priority': 'Medium'
                })
        
        return recommendations
    
    def predict_for_replenishment_decision(self, product_key: int, current_data: pd.DataFrame) -> Dict:
        """
        Make a specific prediction for replenishment decision
        
        Args:
            product_key: Product to predict for
            current_data: Current product data
            
        Returns:
            Dictionary with prediction and business recommendation
        """
        if product_key not in current_data['ProductKey'].values:
            raise ValueError(f"Product {product_key} not found in current data")
        
        product_data = current_data[current_data['ProductKey'] == product_key]
        prediction = self.predict_discontinuation(product_data)
        
        if len(prediction) == 0:
            raise ValueError("Could not generate prediction for this product")
        
        result = prediction.iloc[0]
        
        # Business decision logic
        prob = result['discontinuation_probability']
        risk = result['risk_category']
        confidence = result['confidence_level']
        
        if prob >= 0.8 and confidence == 'High':
            decision = "DO NOT RESTOCK"
            reason = f"Very high discontinuation probability ({prob:.1%}) with high confidence"
        elif prob >= 0.6:
            decision = "RESTOCK MINIMAL QUANTITIES"
            reason = f"High discontinuation probability ({prob:.1%}) - minimize inventory risk"
        elif prob <= 0.3 and confidence == 'High':
            decision = "RESTOCK NORMALLY"
            reason = f"Low discontinuation probability ({prob:.1%}) with high confidence"
        else:
            decision = "RESTOCK WITH CAUTION"
            reason = f"Moderate discontinuation probability ({prob:.1%}) or low confidence"
        
        return {
            'ProductKey': product_key,
            'discontinuation_probability': prob,
            'risk_category': risk,
            'confidence_level': confidence,
            'replenishment_decision': decision,
            'reasoning': reason,
            'last_updated': pd.Timestamp.now().isoformat()
        }
    
    def batch_predict_for_replenishment(self, products_to_check: List[int], 
                                      current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate replenishment recommendations for a batch of products
        
        Args:
            products_to_check: List of ProductKeys to evaluate
            current_data: Current data for all products
            
        Returns:
            DataFrame with replenishment recommendations
        """
        results = []
        
        for product_key in products_to_check:
            try:
                result = self.predict_for_replenishment_decision(product_key, current_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for product {product_key}: {str(e)}")
                results.append({
                    'ProductKey': product_key,
                    'discontinuation_probability': None,
                    'risk_category': 'Unknown',
                    'confidence_level': 'None',
                    'replenishment_decision': 'ERROR - MANUAL REVIEW',
                    'reasoning': str(e),
                    'last_updated': pd.Timestamp.now().isoformat()
                })
        
        return pd.DataFrame(results)
    
    def model_health_check(self) -> Dict[str, Any]:
        """Perform basic health check on loaded model"""
        if self.model is None:
            return {'status': 'ERROR', 'message': 'No model loaded'}
        
        health = {
            'status': 'OK',
            'model_type': self.model_name,
            'feature_count': len(self.feature_columns),
            'model_loaded': True,
            'last_check': pd.Timestamp.now().isoformat()
        }
        
        # Test prediction capability
        try:
            # Create dummy data for testing
            test_data = pd.DataFrame({col: [0] for col in self.feature_columns})
            _ = self.model.predict_proba(test_data)
            health['prediction_capability'] = 'OK'
        except Exception as e:
            health['status'] = 'WARNING'
            health['prediction_capability'] = f'ERROR: {str(e)}'
        
        return health