# Introduction to Machine Learning and Data Science

## Repository Link
https://github.com/cloudbasti/dataScienceCourse

## Project Description

This project aims to predict bakery sales based on historical sales data for various products, incorporating weather data, events such as Kiel Week, and additional features.

## How to Run the Code

### Presteps
1. Make sure you are in the DataScienceCourse directory (in terminal)
2. Run the startup script:
   ```
   python startup_script.py
   ```

### Training
1. Regression model:
   ```
   python 2_BaselineModel/MultipleLinearRegression.py
   ```
2. Neural Network:
   ```
   python 3_Model/neural_simple.py
   ```
3. Learning Rate optimization:
   ```
   python 3_Model/learningRate.py
   ```

## Project Description

This project aims to predict bakery sales based on historical sales data for various products, incorporating weather data, events such as Kiel Week, and additional features.

# Model Performance Analysis

## Introduction

This project demonstrates the successful implementation of advanced regression models for precise sales forecasting. Our approach uses both traditional machine learning methods and deep learning techniques to capture complex data patterns.

The baseline model using linear regression achieved strong performance with an R-squared value of 0.860 and a Root Mean Squared Error (RMSE) of 48.67, providing a solid foundation for our forecasting capabilities.

Building on this foundation, we implemented a neural network architecture that further improved prediction accuracy, consistently achieving R-squared values between 0.88 and 0.89. This represents a significant improvement over the baseline model and highlights the neural network's ability to capture more complex, non-linear relationships in the data.

### Best Model: Neural Network
* **R-squared:** 0.887
* **RMSE:** 43.72
* **Overall MAPE:** 19.89%
* **MAPE by Product Category:**
  * Product 1: 18.46%
  * Product 2: 13.79%
  * Product 3: 18.90%
  * Product 4: 26.45%
  * Product 5: 16.71%
  * Product 6: 52.50%

### Technical Details
* **Architecture:** 128 → 64 → 32 → 1 neurons
* **Hyperparameters:**
  * Batch size: 32
  * Learning rate: 0.000587
  * Epochs: 50 with Early Stopping
* **Regularization:** L2(0.01) with Dropout layers (30%, 30%, 12%)

### Neural Network Performance by Product Category:
* **Product 1:**
  * R-squared: 0.347
  * RMSE: 34.23
  * MAPE: 18.46%
* **Product 2:**
  * R-squared: 0.799
  * RMSE: 56.82
  * MAPE: 13.79%
* **Product 3:**
  * R-squared: 0.706
  * RMSE: 41.07
  * MAPE: 18.90%
* **Product 4:**
  * R-squared: -0.253
  * RMSE: 29.61
  * MAPE: 26.45%
* **Product 5:**
  * R-squared: 0.641
  * RMSE: 52.82
  * MAPE: 16.71%
* **Product 6:**
  * R-squared: 0.321
  * RMSE: 25.50
  * MAPE: 52.50%

## Documentation
1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/README.md)**
3. **[Baseline Model](2_BaselineModel/README.md)**
4. **[Model Definition and Evaluation](3_Model/README.md)**
5. **[Presentation](4_Presentation/README.md)**