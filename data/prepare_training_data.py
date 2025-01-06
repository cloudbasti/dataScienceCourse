import pandas as pd
from data_prep import merge_datasets, prepare_features, handle_missing_values


def prepare_training_data():
    """
    Prepare the complete dataset for training by:
    1. Merging all necessary datasets
    2. Preparing features
    3. Handling missing values
    4. Saving the final dataset
    """
    print("Starting data preparation pipeline...")

    # Step 1: Merge datasets
    print("\nStep 1: Merging datasets...")
    df_merged = merge_datasets()
    print(f"Merged data shape: {df_merged.shape}")

    # Step 2: Prepare features
    print("\nStep 2: Preparing features...")
    df_featured = prepare_features(df_merged)
    print(f"Featured data shape: {df_featured.shape}")

    # Step 3: Handle missing values
    print("\nStep 3: Handling missing values...")
    df_cleaned = handle_missing_values(df_featured)
    print(f"Final data shape: {df_cleaned.shape}")

    # Save the final dataset
    output_path = "data/data_for_training.csv"
    print(f"\nSaving prepared data to {output_path}")
    df_cleaned.to_csv(output_path, index=False)

    return df_cleaned


if __name__ == "__main__":
    df = prepare_training_data()

    # Print some basic statistics about the final dataset
    print("\nFinal Dataset Statistics:")
    print("-" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values by column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
