import pandas as pd
import numpy as np


def merge_and_analyze():
    """
    Perform the merge and analyze the weather codes right after merging
    """
    # Load all auxiliary datasets
    test_df = pd.read_csv("data/test.csv")
    weather = pd.read_csv("data/wetter_imputed.csv")
    kiwo = pd.read_csv("data/kiwo.csv")
    school_holidays = pd.read_csv("data/school_holidays.csv")
    public_holidays = pd.read_csv("data/bank_holidays.csv")

    # Convert dates to datetime in all dataframes
    test_df['Datum'] = pd.to_datetime(test_df['Datum'])
    weather['Datum'] = pd.to_datetime(weather['Datum'])
    kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])
    school_holidays['Datum'] = pd.to_datetime(school_holidays['Datum'])
    public_holidays['Datum'] = pd.to_datetime(public_holidays['Datum'])

    # Start with test_df and merge all other datasets
    df = test_df.copy()

    print("Before merge - shape:", df.shape)

    # Merge weather data
    df = pd.merge(df, weather, on='Datum', how='left')
    print("After weather merge - shape:", df.shape)

    # Store the original weather codes
    df['original_wettercode'] = df['Wettercode']

    # Try numeric conversion
    df['numeric_wettercode'] = pd.to_numeric(df['Wettercode'], errors='coerce')

    # Create mask for invalid codes
    mask = (df['numeric_wettercode'] >= 0) & (df['numeric_wettercode'] <= 99)
    invalid_entries = df[~mask].copy()

    # Save problematic entries
    output_file = "data/merge_weather_problems.csv"
    if len(invalid_entries) > 0:
        print(f"\nFound {len(invalid_entries)
                         } problematic entries after merge")

        # Add analysis columns
        invalid_entries['is_empty'] = invalid_entries['Wettercode'].astype(
            str).str.strip() == ''
        invalid_entries['is_null'] = invalid_entries['Wettercode'].isna()

        # Select relevant columns
        analysis_cols = ['Datum', 'Wettercode', 'original_wettercode',
                         'numeric_wettercode', 'is_empty', 'is_null']
        invalid_entries[analysis_cols].to_csv(output_file, index=True)

        print("\nUnique problematic values:")
        print(invalid_entries['Wettercode'].value_counts().head())

        print("\nSample of problematic entries:")
        print(invalid_entries[analysis_cols].head())
    else:
        print("\nNo problematic entries found after merge")


if __name__ == "__main__":
    merge_and_analyze()
