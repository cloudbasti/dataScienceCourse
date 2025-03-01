import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Add parent directory to path so we can import from data.TrainingPreparation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the needed functions directly
try:
    from data.TrainingPreparation.data_prep import prepare_features, merge_datasets
    print("Successfully imported functions from data_prep.py")
except ImportError as e:
    print(f"Error importing from data_prep.py: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")


def analyze_csv_files():
    """
    Analyzes CSV files and prints sample counts, date ranges, and product distributions
    """
    files = [
        'data/OriginalData/umsatzdaten_gekuerzt.csv', 
        'data/OriginalData/wetter.csv', 
        'data/OriginalData/kiwo.csv'
    ]
    
    print("Dataset Overview - Sample Counts:")
    print("---------------------------------")
    
    for file in files:
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Get and print the number of rows
            num_rows = len(df)
            print(f"{file}: {num_rows} samples")
            
            # Analyze date ranges - assume column is either 'Datum' or 'date'
            date_col = 'Datum' if 'Datum' in df.columns else 'date'
            
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Get overall date range
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            print(f"Date Range: {min_date.date()} to {max_date.date()}")
            
            # For umsatzdaten file, analyze date ranges per product category
            if 'umsatzdaten_gekuerzt.csv' in file and 'Warengruppe' in df.columns:
                print("Date Ranges by Product Category:")
                for category in sorted(df['Warengruppe'].unique()):
                    category_df = df[df['Warengruppe'] == category]
                    cat_min_date = category_df[date_col].min()
                    cat_max_date = category_df[date_col].max()
                    print(f"Warengruppe {category}: {cat_min_date.date()} to {cat_max_date.date()}")
            
            # If this is the umsatzdaten file, analyze product categories
            if 'umsatzdaten_gekuerzt.csv' in file and 'Warengruppe' in df.columns:
                print("Product Category Distribution (Warengruppe):")
                print("---------------------------------")
                
                # Count samples per product category
                category_counts = df['Warengruppe'].value_counts().sort_index()
                
                # Print each category and its count
                for category, count in category_counts.items():
                    print(f"Warengruppe {category}: {count} samples")
                
                # Print total samples as verification
                print(f"Total: {category_counts.sum()} samples")
                print("---------------------------------")
            
        except Exception as e:
            print(f"{file}: Error processing file - {str(e)}")
    
    print("---------------------------------")

def analyze_missing_values():
    """
    Analyzes missing values in all CSV files.
    Reports only the total count of missing values per column.
    """
    files = [
        'data/OriginalData/umsatzdaten_gekuerzt.csv', 
        'data/OriginalData/wetter.csv', 
        'data/OriginalData/kiwo.csv'
    ]
    
    print("\nMissing Values Analysis:")
    print("---------------------------------")
    
    for file in files:
        try:
            # Extract filename without path for cleaner output
            filename = file.split('/')[-1]
            
            # Read the CSV file
            df = pd.read_csv(file)
            
            print(f"File: {filename}")
            print(f"Total rows: {len(df)}")
            
            # Initialize total missing for this file
            total_missing_in_file = 0
            columns_with_missing = {}
            
            # Check each column for missing values
            for column in df.columns:
                missing_count = df[column].isna().sum()
                total_missing_in_file += missing_count
                
                if missing_count > 0:
                    columns_with_missing[column] = missing_count
                    print(f"  {column}: {missing_count} missing values")
            
            # Print summary for this file
            print(f"Summary for {filename}:")
            print(f"  Total missing values: {total_missing_in_file}")
            
        except Exception as e:
            print(f"{file}: Error analyzing missing values - {str(e)}")
    
    print("---------------------------------")

def analyze_turnover_distribution(df):
    """
    Analyzes turnover distribution across different feature groups and prints tabular results.
    
    Args:
        df: DataFrame containing the merged and feature-prepared data
    """
    print("\nTurnover Distribution Analysis")
    print("==============================\n")
    
    comparison_results = []
    
    # 1. Weekday vs Weekend analysis
    weekday_turnover = df[df['is_weekend'] == 'Weekday']['Umsatz'].mean()
    weekend_turnover = df[df['is_weekend'] == 'Weekend']['Umsatz'].mean()
    
    comparison_results.append({
        'Comparison': 'Weekday vs Weekend',
        'Group 1': 'Weekday',
        'Turnover 1': weekday_turnover,
        'Group 2': 'Weekend',
        'Turnover 2': weekend_turnover,
        'Difference': weekend_turnover - weekday_turnover,
        'Ratio': weekend_turnover / weekday_turnover if weekday_turnover > 0 else float('nan')
    })
    
    # 2. New Year's Eve vs Regular Days
    nye_turnover = df[df['is_nye'] == True]['Umsatz'].mean()
    regular_turnover = df[df['is_nye'] == False]['Umsatz'].mean()
    
    comparison_results.append({
        'Comparison': 'New Year\'s Eve vs Regular Days',
        'Group 1': 'New Year\'s Eve',
        'Turnover 1': nye_turnover,
        'Group 2': 'Regular Days',
        'Turnover 2': regular_turnover,
        'Difference': nye_turnover - regular_turnover,
        'Ratio': nye_turnover / regular_turnover if regular_turnover > 0 else float('nan')
    })
    
    # 3. Last Day of Month vs Other Days
    last_day_turnover = df[df['is_last_day_of_month'] == True]['Umsatz'].mean()
    other_days_turnover = df[df['is_last_day_of_month'] == False]['Umsatz'].mean()
    
    comparison_results.append({
        'Comparison': 'Last Day of Month vs Other Days',
        'Group 1': 'Last Day of Month',
        'Turnover 1': last_day_turnover,
        'Group 2': 'Other Days',
        'Turnover 2': other_days_turnover,
        'Difference': last_day_turnover - other_days_turnover,
        'Ratio': last_day_turnover / other_days_turnover if other_days_turnover > 0 else float('nan')
    })
    
    # 4. July/August vs Other Months
    summer_months = df['Datum'].dt.month.isin([7, 8])
    summer_turnover = df[summer_months]['Umsatz'].mean()
    other_months_turnover = df[~summer_months]['Umsatz'].mean()
    
    comparison_results.append({
        'Comparison': 'July/August vs Other Months',
        'Group 1': 'July/August',
        'Turnover 1': summer_turnover,
        'Group 2': 'Other Months',
        'Turnover 2': other_months_turnover,
        'Difference': summer_turnover - other_months_turnover,
        'Ratio': summer_turnover / other_months_turnover if other_months_turnover > 0 else float('nan')
    })
    
    # Create a DataFrame from results for pretty printing
    result_df = pd.DataFrame(comparison_results)
    
    # Format numeric columns to show 2 decimal places
    for col in ['Turnover 1', 'Turnover 2', 'Difference', 'Ratio']:
        result_df[col] = result_df[col].map(lambda x: f"{x:.2f}")
    
    # Print the table
    print(result_df.to_string(index=False))
    
    # Additional analysis: Print detailed stats per product group for weekday vs weekend
    print("\nDetailed Analysis by Product Group (Warengruppe)")
    print("==============================================\n")
    
    # Analyze each comparison by product group
    for product_group in sorted(df['Warengruppe'].unique()):
        product_df = df[df['Warengruppe'] == product_group]
        
        print(f"\nProduct Group (Warengruppe) {product_group}")
        print("-" * 40)
        
        product_results = []
        
        # 1. Weekday vs Weekend
        weekday_turnover = product_df[product_df['is_weekend'] == 'Weekday']['Umsatz'].mean()
        weekend_turnover = product_df[product_df['is_weekend'] == 'Weekend']['Umsatz'].mean()
        
        product_results.append({
            'Comparison': 'Weekday vs Weekend',
            'Group 1': 'Weekday',
            'Turnover 1': weekday_turnover,
            'Group 2': 'Weekend',
            'Turnover 2': weekend_turnover,
            'Difference': weekend_turnover - weekday_turnover,
            'Ratio': weekend_turnover / weekday_turnover if weekday_turnover > 0 else float('nan')
        })
        
        # 2. July/August vs Other Months
        summer_months = product_df['Datum'].dt.month.isin([7, 8])
        summer_turnover = product_df[summer_months]['Umsatz'].mean()
        other_months_turnover = product_df[~summer_months]['Umsatz'].mean()
        
        product_results.append({
            'Comparison': 'July/August vs Other Months',
            'Group 1': 'July/August',
            'Turnover 1': summer_turnover,
            'Group 2': 'Other Months',
            'Turnover 2': other_months_turnover,
            'Difference': summer_turnover - other_months_turnover,
            'Ratio': summer_turnover / other_months_turnover if other_months_turnover > 0 else float('nan')
        })
        
        # Create a DataFrame from results for pretty printing
        product_result_df = pd.DataFrame(product_results)
        
        # Format numeric columns to show 2 decimal places
        for col in ['Turnover 1', 'Turnover 2', 'Difference', 'Ratio']:
            product_result_df[col] = product_result_df[col].map(lambda x: f"{x:.2f}")
        
        # Print the table
        print(product_result_df.to_string(index=False))

def merge_and_save_data():
    """
    Merges datasets and adds features, then saves to a CSV file in the current directory.
    """
    try:
        # Merge datasets
        print("\nMerging datasets...")
        merged_data = merge_datasets()
        print(f"Merged dataset shape: {merged_data.shape}")
        
        # Add features
        print("Adding features...")
        featured_data = prepare_features(merged_data)
        print(f"Featured dataset shape: {featured_data.shape}")
        
        # Save to current directory
        output_path = "merged_featured_data.csv"
        featured_data.to_csv(output_path, index=False)
        print(f"Saved dataset to {output_path}")
        
        return featured_data
    except Exception as e:
        print(f"Error in merge_and_save_data: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        return None



# Call the functions if this script is run directly
if __name__ == "__main__":
    analyze_csv_files()
    analyze_missing_values()
    
    # Added the merge and save functionality call here
    print("\nMerging and preparing the dataset...")
    merged_featured_data = merge_and_save_data()
    
    # Run the turnover distribution analysis on the prepared data
    if merged_featured_data is not None:
        print("\nRunning turnover distribution analysis...")
        analyze_turnover_distribution(merged_featured_data)
    else:
        print("Error: Could not run turnover analysis because data preparation failed.")