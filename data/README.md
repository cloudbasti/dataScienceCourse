# Data Folder Overview

This folder contains all data-related files and scripts for the sales prediction project. Below is an overview of each subfolder and its purpose.

## Folder Structure

- [OriginalData](#originaldata) - Contains raw, unprocessed datasets
- [HolidayData](#holidaydata) - Scripts and data for bank and school holidays
- [WeatherImputation](#weatherimputation) - Scripts for weather data imputation
- [TrainingPreparation](#trainingpreparation) - Code for data merging and feature preparation
- [SubmissionPreparation](#submissionpreparation) - Scripts for test data preparation
- [SubmissionFiles](#submissionfiles) - Final files for model submission

## Detailed Description

### OriginalData

The folder [`OriginalData`](./OriginalData) contains the original raw data files before any processing or transformation. These serve as the starting point for our data pipeline.

Files:
- [`kiwo.csv`](./OriginalData/kiwo.csv) - Original sales data
- [`wetter.csv`](./OriginalData/wetter.csv) - Original weather data
- [`umsatzdaten_gekuerzt.csv`](./OriginalData/umsatzdaten_gekuerzt.csv) - Shortened revenue data

### HolidayData

The folder [`HolidayData`](./HolidayData) contains scripts and generated CSV files for bank holidays and school holidays, which are used as features in our prediction models.

Files:
- [`create_bank_holidays.py`](./HolidayData/create_bank_holidays.py) - Script to generate bank holiday data
- [`create_school_holidays.py`](./HolidayData/create_school_holidays.py) - Script to generate school holiday data
- [`bank_holidays.csv`](./HolidayData/bank_holidays.csv) - Generated bank holiday dataset
- [`school_holidays.csv`](./HolidayData/school_holidays.csv) - Generated school holiday dataset

### WeatherImputation

The folder [`WeatherImputation`](./WeatherImputation) contains scripts for handling missing weather data through imputation techniques. The first Script `Wetter_Imputation.py` acts on the `wetter.csv` file wheras `Final_wetter_imputation.py` contains two functions whcih can be imported. One of these consumes a Dataframe and does imputation on this and the other does the imputation on a final csv file for submission. One would indeed need only one script or service containing one or two handlers, this should be simplified later on.

Files:
- [`Wetter_Imputation.py`](./WeatherImputation/Wetter_Imputation.py) - Initial script that examines the raw weather data and performs imputation
- [`Final_wetter_imputation.py`](./WeatherImputation/Final_wetter_imputation.py) - Used after merging files to perform additional imputation for any new missing values
- [`wetter_imputed.csv`](./WeatherImputation/wetter_imputed.csv) - Weather data after imputation

### TrainingPreparation

The folder [`TrainingPreparation`](./TrainingPreparation) contains reusable code for data preparation and feature engineering.

Files:
- [`data_prep.py`](./TrainingPreparation/data_prep.py) - Contains functions for merging datasets, handling missing values, and adding/preparing features
- [`merged_data.csv`](./TrainingPreparation/merged_data.csv) - Combined dataset ready for training

### SubmissionPreparation

The folder [`SubmissionPreparation`](./SubmissionPreparation) contains scripts for preparing new test data for model predictions.

Files:
- [`prepareTestData.py`](./SubmissionPreparation/prepareTestData.py) - Script for merging and preparing test datasets, including temperature imputation
- [`prepared_test_data.csv`](./SubmissionPreparation/prepared_test_data.csv) - Prepared test data
- [`test_data_final_after_imputation.csv`](./SubmissionPreparation/test_data_final_after_imputation.csv) - Final test data after all imputations
- [`test.csv`](./SubmissionPreparation/test.csv) - Raw test file

### SubmissionFiles

The folder [`SubmissionFiles`](./SubmissionFiles) contains the final files used for model submission.

Files:
- [`Network_Submission.csv`](./SubmissionFiles/Network_Submission.csv) - Neural network model submission
- [`Regression_Submission.csv`](./SubmissionFiles/Regression_Submission.csv) - Regression model submission