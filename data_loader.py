"""
Data loading and validation module
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle data loading and basic validation"""
    
    def __init__(self, product_details_path: Path, catalogue_path: Path):
        self.product_details_path = product_details_path
        self.catalogue_path = catalogue_path
        self.product_details = None
        self.catalogue_data = None
        self.merged_data = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both datasets and perform basic validation"""
        logger.info("Loading product details...")
        self.product_details = pd.read_csv(self.product_details_path)
        
        logger.info("Loading catalogue discontinuation data...")
        self.catalogue_data = pd.read_csv(self.catalogue_path)
        for df_name, df in [("product_details", self.product_details), ("catalogue_data", self.catalogue_data)]:
            for col in df.select_dtypes("int64").columns:
                df[col] = df[col].astype("int32")
            for col in df.select_dtypes("float64").columns:
                df[col] = df[col].astype("float32")
            logger.info(f"Optimized dtypes for {df_name}")

        self._validate_data()
        return self.product_details, self.catalogue_data
    
    def _validate_data(self):
        """Perform basic data validation"""
        logger.info("Validating data...")
        
        # Check if files were loaded successfully
        if self.product_details is None or self.catalogue_data is None:
            raise ValueError("Data files could not be loaded")
        
        # Check for required columns
        required_product_cols = ['ProductKey', 'Supplier', 'HierarchyLevel1', 
                               'HierarchyLevel2', 'DIorDOM', 'Seasonal']
        required_catalogue_cols = ['CatEdition', 'SpringSummer', 'ProductKey', 
                                 'WeeksOut', 'Status', 'SalePriceIncVAT', 
                                 'ForecastPerWeek', 'ActualsPerWeek', 'DiscontinuedTF']
        
        missing_product_cols = set(required_product_cols) - set(self.product_details.columns)
        missing_catalogue_cols = set(required_catalogue_cols) - set(self.catalogue_data.columns)
        
        if missing_product_cols:
            raise ValueError(f"Missing columns in product details: {missing_product_cols}")
        
        if missing_catalogue_cols:
            raise ValueError(f"Missing columns in catalogue data: {missing_catalogue_cols}")
        
        
        # Log basic statistics
        logger.info(f"Product details shape: {self.product_details.shape}")
        logger.info(f"Catalogue data shape: {self.catalogue_data.shape}")
        logger.info(f"Unique products in details: {self.product_details['ProductKey'].nunique()}")
        logger.info(f"Unique products in catalogue: {self.catalogue_data['ProductKey'].nunique()}")
        
    def merge_datasets(self) -> pd.DataFrame:
        """Merge the datasets and return combined dataframe"""
        logger.info("Merging datasets...")
        
        # Merge on ProductKey
        self.merged_data = self.catalogue_data.merge(
            self.product_details, 
            on='ProductKey', 
            how='left'
        )
        
        for col in self.merged_data.select_dtypes("int64").columns:
            self.merged_data[col] = self.merged_data[col].astype("int32")
        for col in self.merged_data.select_dtypes("float64").columns:
            self.merged_data[col] = self.merged_data[col].astype("float32")


        # Check for products that didn't match
        unmatched = self.merged_data['Supplier'].isnull().sum()
        if unmatched > 0:
            logger.warning(f"Found {unmatched} catalogue entries without matching product details")
        
        logger.info(f"Merged dataset shape: {self.merged_data.shape}")
        return self.merged_data
    
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if self.merged_data is None:
            self.merge_datasets()
        
        summary = {
            'total_records': len(self.merged_data),
            'unique_products': self.merged_data['ProductKey'].nunique(),
            'uniqueDIorDOM': self.merged_data['DIorDOM'].nunique(),
            'unoquestatus': self.merged_data['Status'].nunique(),
            'catalogue_editions': sorted(self.merged_data['CatEdition'].unique()),
            'weeks_out_range': {
                'min': self.merged_data['WeeksOut'].min(),
                'max': self.merged_data['WeeksOut'].max()
            },
            'discontinuation_rate': self.merged_data['DiscontinuedTF'].mean(),
            'status_distribution': self.merged_data['Status'].value_counts().to_dict(),
            'seasonal_products': self.merged_data['Seasonal'].sum(),
            'import_vs_domestic': self.merged_data['DIorDOM'].value_counts().to_dict(),
            'missing_values': self.merged_data.isnull().sum().to_dict()
        }
        
        return summary