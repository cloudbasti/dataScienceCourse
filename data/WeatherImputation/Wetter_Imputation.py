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


def impute_data(df):
    """
    Impute missing weather codes and Bewoelkung values.
    """
    # Create a copy of the dataframe
    df_imputed = df.copy()

    # Add month column for analysis
    df_imputed['month'] = df_imputed['Datum'].dt.month

    # Print initial analysis
    # print_missing_analysis(df_imputed, "Before imputation:")

    # First handle Weather Codes
    print("\nStarting Weather Code imputation...")

    # Store initial distributions
    initial_distributions = {}
    for month in range(1, 13):
        month_data = df_imputed[df_imputed['month']
                                == month]['Wettercode'].value_counts()
        initial_distributions[month] = month_data

    # For each month, calculate probability distribution of codes
    for month in range(1, 13):
        # Get monthly data
        month_mask = df_imputed['month'] == month
        month_data = df_imputed[month_mask]

        # Calculate probability distribution of weather codes for this month
        code_counts = month_data['Wettercode'].value_counts()
        total_valid = code_counts.sum()
        code_probs = code_counts / total_valid

        # Find missing values for this month
        missing_mask = month_mask & df_imputed['Wettercode'].isna()
        n_missing = missing_mask.sum()

        if n_missing > 0:
            # Calculate how many times each code should be used
            n_codes = {code: int(prob * n_missing)
                       for code, prob in code_probs.items()}
            # Add remaining codes randomly based on probabilities
            remaining = n_missing - sum(n_codes.values())
            if remaining > 0:
                additional_codes = np.random.choice(
                    code_counts.index,
                    size=remaining,
                    p=code_probs.values
                )
                for code in additional_codes:
                    n_codes[code] = n_codes.get(code, 0) + 1

            # Create list of codes to insert
            codes_to_insert = []
            for code, count in n_codes.items():
                codes_to_insert.extend([code] * count)

            # Shuffle the codes
            np.random.shuffle(codes_to_insert)

            # Insert the codes
            df_imputed.loc[missing_mask, 'Wettercode'] = codes_to_insert

    # Print distributions before and after imputation
    """ print("\nDistributions Before and After Weather Code Imputation:")
    for month in range(1, 13):
        print(f"\nMonth {month}:")
        print("-" * 60)

        # Get before and after distributions
        before_dist = initial_distributions[month]
        after_dist = df_imputed[df_imputed['month']
                                == month]['Wettercode'].value_counts()

        # Combine into a DataFrame for side-by-side comparison
        comparison = pd.DataFrame({
            'Before': before_dist,
            'After': after_dist
        }).fillna(0).astype(int)

        # Add a difference column
        comparison['Difference'] = comparison['After'] - comparison['Before']

        # Sort by 'After' values to maintain same order as your example
        comparison = comparison.sort_values('After', ascending=False)

        print(comparison.head(5)) """

    # Now handle Bewoelkung imputation
    print("\nStarting Bewoelkung imputation...")

    # Calculate median Bewoelkung and round to whole number
    median_bewoelkung = round(df_imputed['Bewoelkung'].median())
    #print(f"Median Bewoelkung value used for imputation: {median_bewoelkung}")

    # Print Bewoelkung distribution before imputation
    """ print("\nBewoelkung distribution before imputation:")
    print(df_imputed['Bewoelkung'].describe()) """

    # Count missing values before Bewoelkung imputation
    missing_before = df_imputed['Bewoelkung'].isna().sum()

    # Impute missing values with rounded median
    df_imputed['Bewoelkung'] = df_imputed['Bewoelkung'].fillna(
        median_bewoelkung)

    # Print Bewoelkung distribution after imputation
    """ print("\nBewoelkung distribution after imputation:")
    print(df_imputed['Bewoelkung'].describe()) """

    # Count missing values after imputation
    #missing_after = df_imputed['Bewoelkung'].isna().sum()
    """ print(f"\nNumber of Bewoelkung values imputed: {
          missing_before - missing_after}") """

    # Print final analysis after both imputations
    #print("\nAfter both Weather Code and Bewoelkung imputation:")
    #print_missing_analysis(df_imputed)

    # Remove the month column before saving
    df_imputed = df_imputed.drop('month', axis=1)

    # Save the imputed data
    df_imputed.to_csv('data/WeatherImputation/wetter_imputed.csv', index=False)
    print("\nImputed data has been saved to 'wetter_imputed.csv'")

    return df_imputed


if __name__ == "__main__":
    # Read the data
    df = pd.read_csv('data/OriginalData/wetter.csv', parse_dates=['Datum'])

    # Run the imputation
    df_imputed = impute_data(df)
