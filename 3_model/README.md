# Neural Network Sales Prediction Model

This README provides detailed information about the neural network models for sales prediction, covering model architecture, feature engineering, implementation details, and evaluation metrics.

## Related Files

- [Neural_Network.py](./Neural_Network.py) - The primary neural network model implementation
- [Final_Neural_Network.py](./Final_Neural_Network.py) - Final submission version trained on full dataset
- [Learning_Rate_Analysis.py](./Learning_Rate_Analysis.py) - Script for learning rate optimization
- [data folder](../data) - Contains preprocessing in [data_prep.py](../data/data_prep.py)

## Model Selection

The implementation uses a **Deep Neural Network** architecture built with TensorFlow/Keras with the following key characteristics:

- **Multi-layer Architecture**: The model consists of three hidden layers (128 → 64 → 32 neurons) followed by a single output neuron for regression.
- **Regularization Techniques**: The model employs multiple regularization strategies:
  - L2 regularization on all dense layers to prevent overfitting
  - Dropout layers (varying from 0.12 to 0.42) to improve generalization
  - Batch normalization to stabilize training
- **Activation Function**: ReLU (Rectified Linear Unit) is used for all hidden layers
- **Loss Function**: Mean Squared Error (MSE) optimizes for the regression task
- **Optimizer**: Adam optimizer with carefully tuned learning rate

This architecture was chosen for its ability to model complex non-linear relationships between features, particularly the interactions between product categories and environmental variables.

## Feature Engineering

The neural network model employs an extensive feature engineering approach:

### Base Features

- **Weather Variables**: Temperature, cloud cover, weather categories (clear, rain, snow, etc.), and wind categories
- **Temporal Variables**: Day of week, month, season, weekend flags
- **Special Events**: Various holiday indicators and special event flags

### Advanced Feature Engineering

1. **Polynomial Features**: 
   - Temperature is expanded to 3rd degree polynomials (Temp, Temp², Temp³)
   - Interactions between polynomial temperature features and cloud cover

2. **Product-Specific Interactions**:
   - Each base feature is interacted with product-specific indicator variables
   - Creates unique effects for each product category

The final feature set is significantly larger than the baseline linear model, enabling the neural network to capture more complex patterns in the data.

## Hyperparameter Tuning

The neural network implementation includes:

1. **Learning Rate Exploration**:
   - A dedicated script ([Learning_Rate_Analysis.py](./Learning_Rate_Analysis.py)) explores the impact of learning rate
   - Tests a logarithmic range of learning rates from 0.0001 to 0.001
   - Evaluates each learning rate's performance on R², RMSE, and loss metrics
   - Helps identify suitable learning rate values for the model

2. **Model Configuration**:
   - Layer sizes (128, 64, 32 neurons)
   - Dropout rates (0.12 to 0.42)
   - L2 regularization strength (0.01)

3. **Training Parameters**:
   - Batch size of 32
   - Early stopping with patience of 50 epochs
   - Learning rate reduction on plateau

While the model includes these parameters, it's worth noting that they were not fully optimized through systematic grid search or other exhaustive methods. Further optimization could potentially yield improved results.

## Implementation

### Training Pipeline

The implementation follows this workflow:

1. **Data Preparation**:
   - Data merging from multiple sources
   - Weather data imputation
   - Feature creation with `prepare_features()`
   - Missing value handling

2. **Feature Engineering**:
   - Generation of complex features using `create_product_features()`
   - Standardization of input features and target variable

3. **Model Training**:
   - Data splitting into training and validation sets (in evaluation version)
   - Neural network initialization with regularization
   - Training with callbacks for early stopping and learning rate reduction
   - Model evaluation on validation data

4. **Production Version / Submission**:
   - The final implementation ([Final_Neural_Network.py](./Final_Neural_Network.py)) trains on the full dataset
   - Makes predictions on test data
   - Generates submission file

### Key Components

- **Callbacks**: Early stopping and learning rate reduction prevent overfitting
- **Data Scaling**: Features and target variable are standardized
- **Visualization**: Training history plots show loss and metrics over time

## Evaluation Metrics

The model is evaluated using multiple metrics:

1. **R-squared (R²)**:
   - Measures how well the model explains variance in sales
   - Calculated for both overall performance and per product category

2. **Root Mean Squared Error (RMSE)**:
   - Measures prediction error in the original units
   - Useful for understanding absolute error magnitude

3. **Mean Absolute Error (MAE)**:
   - Alternative error measure less sensitive to outliers
   - Used during training for monitoring progress

4. **Mean Absolute Percentage Error (MAPE)**:
   - Measures percentage error
   - Provides insight into relative prediction accuracy

### Performance Results

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

## Comparative Analysis

The neural network model offers several advantages over the baseline linear regression:

1. **Non-linear Modeling**:
   - Captures complex non-linear relationships missed by linear models
   - Utilizes polynomial features and interactions more effectively

2. **Feature Handling**:
   - Manages large feature spaces more efficiently
   - Better utilizes complex interaction terms

3. **Regularization**:
   - Multiple regularization techniques prevent overfitting
   - More robust to noise in the data

4. **Performance Improvements**:
   - Better overall R² and RMSE metrics compared to linear regression
   - More consistent performance across product categories
   - Particularly improved predictions for challenging products (e.g., Product 4)

While the linear model provides excellent interpretability through explicit coefficients, the neural network delivers improved accuracy at the cost of some interpretability. For production usage, the neural network's better predictive performance makes it the preferred model.