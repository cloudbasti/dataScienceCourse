# Baseline Sales Prediction Model

This README provides detailed information about the baseline machine learning model for sales prediction, covering model choice, feature selection, implementation details, and evaluation metrics.

## Related Files

- [MultipleLinearRegression.py](./MultipleLinearRegression.py) - The baseline model implementation
- [Final_Regression.py](./Final_Regression.py) - Production version that trains on full dataset
- [data folder](../data) - Contains preprocessing functionality in [data_prep.py](../data/data_prep.py)

## Choice of Model

### Linear Regression

The model implements a standard Linear Regression approach from scikit-learn for predicting sales ("Umsatz") across different product categories. This choice is justified by its strong interpretability, with clear coefficients that directly show each feature's impact on sales, while serving as an excellent baseline to establish fundamental relationships before exploring more complex algorithms. The model has been enhanced with interaction terms to capture how different features interact with product categories, allowing for product-specific effects, and it handles the moderate-sized dataset efficiently without requiring extensive computational resources.

## Feature Selection

### Base Features

The model uses a comprehensive set of features:

1. **Weather Variables**:
   - Temperature ("Temperatur")
   - Cloud cover ("Bewoelkung")
   - Weather category features (clear, rain, snow, fog, etc.)
   - Wind categories (calm, moderate, strong)

2. **Temporal Variables**:
   - Day of week (Monday through Sunday, one-hot encoded)
   - Month (1-12, one-hot encoded)
   - Season (Winter, Spring, Summer, Autumn)
   - Weekend/weekday flags

3. **Special Events**:
   - Holiday indicators (is_holiday)
   - School holiday indicators (is_school_holiday)
   - Special event: "Kieler Woche" (local festival)
   - New Year's Eve (is_nye)
   - Pre-holiday flag
   - Weekend holiday flag
   - Summer weekend flag
   - Peak summer indicators

### Feature Engineering

The model employs sophisticated feature engineering techniques:

1. **Interaction Features**: 
   - The core of the model is the creation of interaction terms between product categories and all base features
   - Each feature is interacted with product-specific dummy variables, allowing the model to learn different effects of each feature for each product category

2. **One-Hot Encoding**:
   - Product categories are one-hot encoded (prefix: 'is_product_')
   - Categorical features like weekday, month, season, weather conditions are one-hot encoded

3. **Missing Value Handling**:
   - Custom imputation for weather data using `impute_weather_data()`
   - General missing value handling through `handle_missing_values()`

## Implementation

### Data Processing Pipeline

The implementation follows a structured pipeline:

1. **Data Integration**:
   - Multiple datasets are merged using `merge_datasets()`

2. **Data Preparation**:
   - Weather data imputation with custom logic
   - Feature preparation using `prepare_features()`
   - Missing value handling

3. **Feature Engineering**:
   - Interaction features created via `create_interaction_features()`
   - Standard scaling applied to the target variable (Umsatz)

4. **Model Training**:
   - Train-validation split using `split_train_validation()`
   - Linear regression model fitting

5. **Prediction and Evaluation**:
   - Predictions on validation set
   - Inverse scaling to get predictions in original scale
   - Calculation of overall and product-specific metrics

### Model Equations

The model generates product-specific equations in the form:

```
Umsatz = Intercept + Coef1*Feature1 + Coef2*Feature2 + ...
```

These equations are stored in `product_equations` and printed for interpretation, allowing direct analysis of feature impacts on each product category.

## Evaluation

### Metrics

The model is evaluated using:

1. **R-squared (R²)**: 
   - Measures the proportion of variance in sales explained by the model

2. **Root Mean Squared Error (RMSE)**:
   - Measures the average magnitude of prediction errors

### Performance Analysis

The evaluation includes:

1. **Overall Model Performance**: 
   - Aggregate R² and RMSE across all products
   - **Results**: R-squared score of 0.860 with RMSE of 48.66

2. **Product-Specific Performance**:
   - Individual R² and RMSE for each product category
   - **Results**:
     - Product 1: R² = 0.363, RMSE = 33.81
     - Product 2: R² = 0.675, RMSE = 72.27
     - Product 3: R² = 0.705, RMSE = 41.12
     - Product 4: R² = -0.046, RMSE = 27.05
     - Product 5: R² = 0.573, RMSE = 57.56
     - Product 6: R² = 0.286, RMSE = 26.14
   - Allows identification of products that are well-predicted (Product 3) vs. those that may need additional features or different modeling approaches (Product 4)

3. **Feature Importance Analysis**:
   - Product-specific equations reveal which features have the strongest influence on sales for each product
   - Coefficients show the magnitude and direction of each feature's impact

## Production Implementation

The [Final_Regression.py](./Final_Regression.py) file provides the production implementation of the model with the following characteristics:

- **Full Dataset Training**: Unlike the baseline model which splits data for evaluation, the final model trains on the complete dataset to maximize predictive power
- **Test Data Processing**: Includes functionality to process new test data with the same feature engineering pipeline
- **Submission Generation**: Creates a submission file with predictions that can be directly used
- **Streamlined Code**: Removes evaluation components to focus solely on making accurate predictions

The final regression implementation follows best practices by leveraging all available data for the final model while maintaining the same feature engineering approach that proved effective in the baseline evaluations.








