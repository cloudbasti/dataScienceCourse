import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_prep import prepare_features
from data.data_prep import merge_datasets
from data.data_prep import handle_missing_values

def train_product_models(df):
    """
    Train separate models for each product category.
    """
    features = ['Temperatur', 'Bewoelkung', 'Wochentag_encoded', 'is_holiday', 'is_school_holiday']
    product_models = {}
    product_metrics = {}
    product_equations = {}
    
    for product_id in range(1, 7):
        # Filter data for this product
        product_data = df[df['Warengruppe'] == product_id].copy()
        
        # Prepare features and target
        X = product_data[features]
        y = product_data['Umsatz']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store results
        product_models[product_id] = model
        product_metrics[product_id] = {'R2': r2, 'RMSE': rmse}
        
        # Store equation coefficients
        product_equations[product_id] = {
            'intercept': model.intercept_,
            'coefficients': dict(zip(features, model.coef_))
        }
    
    return product_models, product_metrics, product_equations

def print_product_equations(product_equations):
    """
    Print the linear equation for each product.
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
    # Load and prepare data
    df_merged = merge_datasets()
    df_featured, weekday_encoder = prepare_features(df_merged)
    df_cleaned = handle_missing_values(df_featured)
    
    # Train models
    product_models, product_metrics, product_equations = train_product_models(df_cleaned)
    
    # Print results
    print("\nModel Performance by Product Category:")
    for product_id, metrics in product_metrics.items():
        print(f"\nProduct {product_id}:")
        print(f"R-squared: {metrics['R2']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
    
    # Print equations
    print_product_equations(product_equations)
    
    # Example predictions
    print("\nExample Predictions:")
    example_data = pd.DataFrame({
        'Temperatur': [24],
        'Bewoelkung': [3],
        'Wochentag_encoded': [weekday_encoder.transform(['Wednesday'])[0]],
        'is_holiday': [0],
        'is_school_holiday': [1]
    })
    
    for product_id, model in product_models.items():
        predicted_umsatz = model.predict(example_data)[0]
        print(f"\nPredicted turnover for Product {product_id} "
              f"(Wednesday, Temp=24, Clouds=3, Not Holiday): {predicted_umsatz:.2f}")

if __name__ == "__main__":
    main()