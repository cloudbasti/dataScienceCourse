import pandas as pd
import numpy as np


def print_missing_analysis(df, title="Missing Values Analysis for Weather Data:"):
    """Print missing values analysis for all columns"""
    print(f"\n{title}")
    print("-" * 50)
    print(f"Total rows in dataset: {len(df)}")

    for column in ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode']:
        null_count = df[column].isna().sum()
        percentage = (null_count / len(df)) * 100
        print("-" * 50)
        print(f"Column: {column}")
        print(f"Null values: {null_count}")
        print(f"Percentage missing: {percentage:.2f}%")


def impute_bewoelkung(df):
    """
    Impute missing Bewoelkung values with mean value.
    """
    # Create a copy of the dataframe
    df_imputed = df.copy()

    # Print initial analysis
    print_missing_analysis(df_imputed, "Before imputation:")

    # Calculate mean Bewoelkung
    mean_bewoelkung = df_imputed['Bewoelkung'].mean()
    print(f"\nMean Bewoelkung value used for imputation: {
          mean_bewoelkung:.2f}")

    # Count missing values before imputation
    missing_before = df_imputed['Bewoelkung'].isna().sum()

    # Impute missing values with mean
    df_imputed['Bewoelkung'] = df_imputed['Bewoelkung'].fillna(mean_bewoelkung)

    # Count missing values after imputation
    missing_after = df_imputed['Bewoelkung'].isna().sum()

    print(f"\nNumber of values imputed: {missing_before - missing_after}")

    # Print final analysis
    print_missing_analysis(df_imputed, "\nAfter imputation:")

    # Print distribution statistics before and after
    print("\nBewoelkung Distribution Statistics:")
    print("\nBefore imputation:")
    print(df['Bewoelkung'].describe())
    print("\nAfter imputation:")
    print(df_imputed['Bewoelkung'].describe())

    # Save the imputed data
    df_imputed.to_csv('wetter_bewoelkung_imputed.csv', index=False)
    print("\nImputed data has been saved to 'wetter_bewoelkung_imputed.csv'")

    return df_imputed


if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('data/wetter.csv', parse_dates=['Datum'])

    # Run the imputation
    df_imputed = impute_bewoelkung(df)
