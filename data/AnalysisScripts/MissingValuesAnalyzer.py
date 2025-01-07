import pandas as pd
import numpy as np


def analyze_missing_values(df):
    # Define possible missing value indicators
    missing_indicators = ['', 'nan', 'NaN', 'NA', 'null', 'NULL', None]

    # Create a copy of the dataframe
    df_check = df.copy()

    # Replace all missing indicators with NaN
    df_check = df_check.replace(missing_indicators, np.nan)

    # Calculate missing values for each column
    missing_values = df_check.isnull().sum()

    # Calculate percentage of missing values
    missing_percentage = (df_check.isnull().sum() / len(df_check)) * 100

    # Get data type for each column
    dtypes = df_check.dtypes

    # Create a summary DataFrame
    missing_summary = pd.DataFrame({
        'Data Type': dtypes,
        'Missing Values': missing_values,
        'Percentage Missing': missing_percentage.round(2)
    })

    return missing_summary


def analyze_domain_missing(df):
    """Analyze domain-specific patterns of missing values"""
    # Time-based patterns
    df['year'] = df['Datum'].dt.year
    df['month'] = df['Datum'].dt.month
    df['day_of_week'] = df['Datum'].dt.dayofweek

    # Analyze missing values by time periods
    print("\nMissing values by year:")
    yearly_missing = df.groupby('year').apply(
        lambda x: x.isnull().sum()
    )
    print(yearly_missing)

    print("\nMissing values by month:")
    monthly_missing = df.groupby('month').apply(
        lambda x: x.isnull().sum()
    )
    print(monthly_missing)

    print("\nMissing values by day of week:")
    daily_missing = df.groupby('day_of_week').apply(
        lambda x: x.isnull().sum()
    )
    print(daily_missing)

    # Analyze value ranges when data is missing
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            print(f"\nStatistics for other variables when {
                  column} is missing:")
            missing_stats = df[df[column].isnull()].describe()
            print(missing_stats)


if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('data/wetter.csv', parse_dates=['Datum'])

    # Get the basic missing values analysis
    print("\nMissing Values Summary:")
    print(analyze_missing_values(df))

    # Additional check for potential hidden characters or whitespace
    print("\nUnique values in each column (to check for hidden missing values):")
    for column in df.columns:
        unique_vals = df[column].unique()
        print(f"\n{column}:")
        print(unique_vals[:10])  # Show first 10 unique values

    # Run domain analysis
    analyze_domain_missing(df)
