"""
Feature engineering module for product discontinuation prediction
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to handle feature engineering for the discontinuation prediction model"""
    
    def __init__(self, verbose_shapes: bool = True):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.verbose_shapes = verbose_shapes

    def _print_shape_info(self, df: pd.DataFrame, step_name: str):
        """Helper method to print shape information"""
        if self.verbose_shapes:
            print(f"[{step_name}] Data shape: {df.shape} (rows: {df.shape[0]:,}, columns: {df.shape[1]:,})")
    
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the model"""
        logger.info("Starting feature engineering...")
        print("Columns before feature engineering:", df.columns.tolist())
        
        df_features = df.copy()
        
        # Basic derived features
        df_features = self._create_performance_features(df_features)
        self._print_shape_info(df_features, "After performance features")
        df_features = self._create_temporal_features(df_features)
        self._print_shape_info(df_features, "After temporal features")

        df_features = self._create_categorical_features(df_features)
        self._print_shape_info(df_features, "After categorical features")
        df_features = self._create_interaction_features(df_features)
        self._print_shape_info(df_features, "After interaction features")
        
        logger.info(f"Feature engineering complete. Total features: {df_features.shape[1]}")
        print(f"=== FEATURE ENGINEERING SUMMARY ===")
        print(f"Final shape: {df_features.shape}")
        print(f"Features added: {df_features.shape[1] - df.shape[1]}")
        print(f"=====================================")
        
        return df_features
    
    def _create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance-based features"""
        logger.info("Creating performance features...")
        
        # Forecast accuracy
        df['forecast_accuracy'] = df['ActualsPerWeek'] / (df['ForecastPerWeek'] + 1e-6)
        df['forecast_error'] = df['ForecastPerWeek'] - df['ActualsPerWeek']
        df['forecast_error_abs'] = np.abs(df['forecast_error'])
        
        # Performance indicators
        df['is_overperforming'] = (df['forecast_accuracy'] > 1).astype(int)
        df['is_underperforming'] = (df['forecast_accuracy'] < 0.8).astype(int)
        df['is_low_volume'] = (df['ActualsPerWeek'] < 1).astype(int)
        
        # Log transforms for skewed features
        df['log_actuals'] = np.log1p(df['ActualsPerWeek'])
        df['log_forecast'] = np.log1p(df['ForecastPerWeek'])
        df['log_price'] = np.log1p(df['SalePriceIncVAT'])
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        # Weeks out categories
        df['weeks_out_category'] = pd.cut(
            df['WeeksOut'], 
            bins=[-np.inf, -24, -12, -8, -4, -1],
            labels=['long_term', 'medium_term', 'short_term', 'immediate', 'decision_point'],
            right=True
        )
        
        # Catalogue edition trends
        df['cat_edition_normalized'] = (df['CatEdition'] - df['CatEdition'].min()) / (df['CatEdition'].max() - df['CatEdition'].min())
        
        # Seasonal alignment
        df['seasonal_mismatch'] = ((df['Seasonal'] == 1) & (df['SpringSummer'] == 0)).astype(int)
        
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Creating categorical features...")
        
        # One-hot encode key categorical variables
        
        df = pd.get_dummies(df, columns=['DIorDOM', 'Status'], prefix=['sourcing', 'status'])
        
        # Label encode hierarchical features for potential tree-based models
        for col in ['Supplier', 'HierarchyLevel1', 'HierarchyLevel2']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                seen_categories = set(self.label_encoders[col].classes_)
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: self.label_encoders[col].transform([x])[0] if x in seen_categories else -1
                )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        # Price vs performance
        df['price_performance_ratio'] = df['SalePriceIncVAT'] / (df['ActualsPerWeek'] + 1)
        
        # Seasonal and sourcing interactions
        df['seasonal_spring_summer'] = df['Seasonal'] * df['SpringSummer']
        df['seasonal_domestic'] = df['Seasonal'] * (df['sourcing_DOM'] if 'sourcing_DOM' in df.columns else 0)
        
        # Performance and timing
        df['performance_timing'] = df['forecast_accuracy'] * np.exp(-df['WeeksOut'] / 10)
        
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features at product, supplier, and category level"""
        logger.info("Creating aggregated features...")
        
        # Sort by ProductKey and WeeksOut for proper time series operations
        df_sorted = df.sort_values(['ProductKey', 'WeeksOut'])
        
        # Product-level historical performance
        for group_col in ['ProductKey', 'Supplier_encoded', 'HierarchyLevel1_encoded']:
            if group_col in df_sorted.columns:
                # Rolling averages
                df_sorted[f'{group_col}_avg_actuals'] = df_sorted.groupby(group_col)['ActualsPerWeek'].transform(
                    lambda x: x.expanding().mean()
                )
                df_sorted[f'{group_col}_avg_accuracy'] = df_sorted.groupby(group_col)['forecast_accuracy'].transform(
                    lambda x: x.expanding().mean()
                )
                
                # Historical discontinuation rates
                df_sorted[f'{group_col}_disc_rate'] = df_sorted.groupby(group_col)['DiscontinuedTF'].transform(
                    lambda x: x.expanding().mean()
                )
        
        return df_sorted
    
    def prepare_model_features(self, df: pd.DataFrame, training: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare final feature set for modeling"""
        logger.info("Preparing model features...")
        
        # Define feature columns (excluding target and identifiers)
        exclude_cols = [
            'ProductKey', 'DiscontinuedTF', 'CatEdition', 'SpringSummer',
            'Supplier', 'HierarchyLevel1', 'HierarchyLevel2',
            'ForecastPerWeek', 'ActualsPerWeek', 'SalePriceIncVAT',  # Keep raw versions out
            'weeks_out_category'  # Categorical version
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        df_features = df[feature_cols].copy()
        df_features = df_features.fillna(0)  # Simple strategy - could be improved

        if training:
          df_features = pd.DataFrame(self.scaler.fit_transform(df_features), columns=feature_cols)
        else:
          df_features = pd.DataFrame(self.scaler.transform(df_features), columns=feature_cols)

        
        # Store feature columns for later use
        if training:
            self.feature_columns = feature_cols
        
        logger.info(f"Final feature set: {len(feature_cols)} features")
        return df_features, feature_cols
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for importance analysis"""
        return self.feature_columns.copy() if self.feature_columns else []