# Product Discontinuation Prediction

A machine learning solution to predict which products will be discontinued at the next Store catalogue refresh, helping the replenishment team make intelligent restocking decisions.

## Business Problem

Store refreshes its product catalogue twice annually, discontinuing approximately 30% of products each time. The replenishment team needs to decide whether to restock items, balancing:
- **Lost sales** from stockouts on continuing products
- **Clearance costs** from excess inventory of discontinued products

This is particularly challenging for low-volume products (less than 1 unit/month in some locations).

## Solution Overview

This solution provides a production-ready machine learning pipeline that:
1. Predicts discontinuation probability for any product at any time
2. Categorizes products by risk level (High/Medium/Low Risk)
3. Provides specific replenishment recommendations
4. Handles the business constraints and seasonal patterns

## Project Structure

```
discontinuation-prediction/
├── config.py                 # Configuration settings
├── data_loader.py            # Data loading and validation
├── feature_engineering.py    # Feature creation and preprocessing
├── model_trainer.py          # Model training and evaluation
├── predictor.py              # Production prediction interface
├── main_pipeline.py          # Complete training pipeline
├── requirements.txt          # Python dependencies
├── data/                     # Data files (place CSV files here)
│   ├── ProductDetails.csv
│   └── CatalogueDiscontinuation.csv
├── models/                   # Saved trained models
├── results/                  # Model results and predictions
└── notebooks/               # Jupyter notebooks for exploration
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your data files in the `data/` directory:
- `ProductDetails.csv`
- `CatalogueDiscontinuation.csv`

### 3. Run the Complete Pipeline
```bash
# Train models and make sample predictions
python main_pipeline.py --mode both

# Train only
python main_pipeline.py --mode train

# Predict only (requires existing model)
python main_pipeline.py --mode predict
```

## Key Features

### Feature Engineering
- **Performance Metrics**: Forecast accuracy, sales trends, volume indicators
- **Temporal Features**: Time to decision, seasonal patterns, catalogue trends
- **Product Hierarchy**: Category-level performance aggregations
- **Business Logic**: Risk indicators for low-volume and underperforming products

### Model Selection
Evaluates multiple algorithms:
- Logistic Regression (baseline, interpretable)
- Random Forest (handles feature interactions)
- XGBoost (high performance gradient boosting)
- LightGBM (fast gradient boosting)

Uses time-based validation splits respecting catalogue edition chronology.

### Production Interface
The `DiscontinuationPredictor` class provides:
- Individual product predictions
- Batch processing for replenishment decisions
- Business recommendations with risk categories
- Model health checking

## Usage Examples

### Basic Prediction
```python
from predictor import DiscontinuationPredictor
import pandas as pd

# Load trained model
predictor = DiscontinuationPredictor('models/best_model_xgboost.pkl')

# Make predictions
predictions = predictor.predict_discontinuation(current_product_data)
print(predictions[['ProductKey', 'discontinuation_probability', 'risk_category']])
```

### Replenishment Decision
```python
# Get specific replenishment recommendation
decision = predictor.predict_for_replenishment_decision(
    product_key=12345, 
    current_data=product_data
)
print(f"Decision: {decision['replenishment_decision']}")
print(f"Reason: {decision['reasoning']}")
```

### Batch Processing
```python
# Process multiple products for replenishment team
products_to_check = [12345, 67890, 54321]
recommendations = predictor.batch_predict_for_replenishment(
    products_to_check, 
    current_data
)
```

## Model Performance

The solution optimizes for **ROC AUC** as the primary metric, as it's appropriate for:
- Imbalanced datasets (not all products are discontinued)
- Probability-based decision making
- Business cost trade-offs

Additional metrics tracked:
- **Precision/Recall** for discontinuation class
- **Cross-validation scores** for stability
- **Feature importance** for interpretability

## Business Integration

### Risk Categories
- **High Risk** (≥80% probability): Stop restocking immediately
- **Medium Risk** (60-79%): Reduce stock levels, monitor closely  
- **Low-Medium Risk** (40-59%): Continue with caution
- **Low Risk** (<40%): Normal restocking

### Confidence Levels
- **High Confidence**: Probability >0.9 or <0.1 (clear decision)
- **Medium Confidence**: Clear trend but moderate certainty
- **Low Confidence**: Close to 50/50 - requires human judgment

## Data Requirements

### ProductDetails.csv
- `ProductKey`: Unique identifier
- `Supplier`: Brand/manufacturer 
- `HierarchyLevel1/2`: Product categories
- `DIorDOM`: Domestic vs Imported sourcing
- `Seasonal`: Has distinct seasonal demand

### CatalogueDiscontinuation.csv
- `CatEdition`: Catalogue refresh identifier
- `SpringSummer`: Season flag
- `WeeksOut`: Weeks before/after decision
- `Status`: Current listing status (RI/RO)
- `SalePriceIncVAT`: Product price
- `ForecastPerWeek`: Expected demand
- `ActualsPerWeek`: Historical demand
- `DiscontinuedTF`: Target variable

## Model Limitations & Assumptions

### Current Limitations
1. **Status Field Usage**: Assumes 'Status' (RI/RO) is available as a feature - may need validation
2. **Missing Data**: Simple imputation strategy (fill with 0) - could be improved
3. **Seasonality**: Basic seasonal features - could expand for more complex patterns
4. **Geographic Variation**: Not explicitly modeled at store level

### Business Assumptions
1. Historical patterns predict future discontinuation decisions
2. Feature relationships remain stable across catalogue editions  
3. 30% discontinuation rate is representative going forward
4. Cost of stockouts vs clearance is reflected in business metrics

### Future Improvements
1. **Advanced Time Series**: LSTM/Prophet for seasonal decomposition
2. **Survival Analysis**: Model time-to-discontinuation explicitly  
3. **Ensemble Methods**: Combine multiple model types
4. **Real-time Updates**: Online learning for concept drift
5. **Store-level Modeling**: Geographic demand variation
6. **Cost-sensitive Learning**: Explicit stockout/clearance cost modeling

## Monitoring & Maintenance

### Model Retraining
- Retrain after each catalogue refresh (every 6 months)
- Monitor feature drift and prediction accuracy
- A/B test new model versions

### Health Checks
```python
health = predictor.model_health_check()
print(health['status'])  # OK, WARNING, or ERROR
```

### Performance Monitoring
Track key business metrics:
- Prediction accuracy on actual discontinuation decisions
- Impact on inventory costs (stockouts vs clearance)
- User adoption by replenishment team

## Contributing

For improvements or bug fixes:
1. Add comprehensive logging for debugging
2. Include unit tests for new features  
3. Update documentation for API changes
4. Validate with business stakeholders

## Contact

For questions about model implementation or business requirements, contact @ajay.jeswani73@gmail.com