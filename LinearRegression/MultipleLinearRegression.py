import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.train_split import split_train_validation
from data.data_prep import prepare_features
from data.data_prep import merge_datasets
from data.data_prep import handle_missing_values

def create_interaction_features(df):
    """
    Create interaction terms between products and features.
    This is specific to linear regression modeling.
    """
    df_with_interactions = df.copy()
    
    # Get all weekday and season dummy columns
    weekday_columns = [col for col in df_with_interactions.columns if col.startswith('weekday_')]
    season_columns = [col for col in df_with_interactions.columns if col.startswith('season_')]
    
    base_features = ['Temperatur', 'Bewoelkung', 'is_holiday', 'is_school_holiday', 
                    'KielerWoche', 'is_weekend'] + weekday_columns + season_columns
    
    product_features = []
    interaction_features = []
    
    # Create product indicators first
    for product_id in range(1, 7):
        product_col = f'is_product_{product_id}'
        df_with_interactions[product_col] = (df_with_interactions['Warengruppe'] == product_id).astype(int)
        product_features.append(product_col)
        
        # Create interaction terms
        for feature in base_features:
            interaction_col = f'int_{feature}_p{product_id}'
            df_with_interactions[interaction_col] = df_with_interactions[product_col] * df_with_interactions[feature]
            interaction_features.append(interaction_col)
    
    all_features = product_features + interaction_features
    return df_with_interactions[['Datum'] + all_features + ['Umsatz']], all_features

def prepare_and_predict_umsatz(df):
    """
    Main function that creates a single model with product-specific equations.
    """
    # Add interaction terms for linear regression
    df_with_interactions, feature_columns = create_interaction_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_validation(df_with_interactions, feature_columns)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate overall metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate metrics per product category
    product_metrics = {}
    for product_id in range(1, 7):
        mask = X_test[f'is_product_{product_id}'] == 1
        if mask.any():
            product_r2 = r2_score(y_test[mask], y_pred[mask])
            product_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            product_metrics[f"Product {product_id}"] = {'R2': product_r2, 'RMSE': product_rmse}
    
    # Extract coefficients for each product
    product_equations = {}
    weekday_columns = [col for col in df_with_interactions.columns if col.startswith('weekday_')]
    season_columns = [col for col in df_with_interactions.columns if col.startswith('season_')]
    
    base_features = ['Temperatur', 'Bewoelkung', 'is_holiday', 'is_school_holiday', 
                    'KielerWoche', 'is_weekend'] + weekday_columns + season_columns
    
    for product_id in range(1, 7):
        # Get product indicator coefficient
        intercept = model.intercept_
        product_coef = model.coef_[feature_columns.index(f'is_product_{product_id}')]
        
        # Get interaction coefficients
        feature_coefs = {}
        for feature in base_features:
            interaction_col = f'int_{feature}_p{product_id}'
            coef = model.coef_[feature_columns.index(interaction_col)]
            feature_coefs[feature] = coef
        
        product_equations[product_id] = {
            'intercept': intercept + product_coef,
            'coefficients': feature_coefs
        }
    
    return model, product_equations, r2, rmse, product_metrics, feature_columns

def print_product_equations(product_equations):
    """
    Print the linear equation for each product in a readable format.
    """
    print("\nProduct-specific Linear Equations:")
    for product_id, eq in product_equations.items():
        print(f"\nProduct {product_id}:")
        equation = f"Umsatz = {eq['intercept']:.2f}"
        for feature, coef in eq['coefficients'].items():
            sign = '+' if coef >= 0 else '-'
            equation += f" {sign} {abs(coef):.2f}*{feature}"
        print(equation)

def main():
    # Load and merge data
    df_merged = merge_datasets()
    
    # Prepare features (now includes weekday dummies)
    df_featured = prepare_features(df_merged)
    
    # Handle missing values
    df_cleaned = handle_missing_values(df_featured)
    
    # Get list of season dummy columns from the cleaned data
    season_columns = [col for col in df_cleaned.columns if col.startswith('season_')]
    
    # Train the model
    model, product_equations, r2, rmse, product_metrics, feature_columns = prepare_and_predict_umsatz(df_cleaned)
    
    # Print overall results
    print("\nOverall Model Performance:")
    print(f"R-squared score: {r2:.3f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Print performance by product category
    print("\nModel Performance by Product Category:")
    for category, metrics in product_metrics.items():
        print(f"\n{category}:")
        print(f"R-squared: {metrics['R2']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
    
    # Print product-specific equations
    print_product_equations(product_equations)
    
    

if __name__ == "__main__":
    main()