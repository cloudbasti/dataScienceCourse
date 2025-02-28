# Literature Review

## Overview
This literature review examines prior work on predicting bakery sales using neural network regression, with an emphasis on integrating weather data, historical sales, and event-based factors. The goal is to deepen understanding of the problem domain—sales prediction for six bakery product categories—and identify established approaches. Key questions include: Which models are commonly used? What format does training data take? How much data is typically required? These are not exactly what we do, but they use neural networks for this task, offering valuable insights into applicable techniques.

## Approaches or Solutions
Bakery sales prediction frequently employs neural networks to capture intricate relationships between sales, weather variables (e.g., temperature, precipitation), and events (e.g., holidays, major weather events). Historical sales data uncovers seasonal patterns and trends, while external factors like weather and events account for demand variability. Neural network regression excels in this context due to its capacity to model non-linear interactions and handle diverse, multivariate inputs effectively.

## Summary of Each Work

- **Source 1: "Utilizing Artificial Neural Networks to Predict Demand for Weather-Sensitive Products at Retail Stores"**
  - [Link](https://arxiv.org/abs/1711.08325)
  - **Objective:** Predict sales of 111 weather-sensitive products (including bakery items) at Walmart stores during major weather events.
  - **Methods:** Employs artificial neural networks (ANNs) with regression outputs, trained on historical sales, weather data (e.g., storms, temperature), and event flags. Data is preprocessed into structured tables with daily granularity.
  - **Outcomes:** ANN models reduce forecasting errors compared to linear regression, using 5 years of data (millions of records) across 45 stores, capturing weather-driven demand spikes effectively.
  - **Relation to the Project:** Confirms ANNs as a robust model for weather-influenced bakery sales prediction. Highlights the need for large historical datasets (millions of points) and structured tabular data, though pretrained models aren’t mentioned.

- **Source 2: "Food Sales Prediction with Meteorological Data — A Case Study of a Japanese Chain Supermarket"**
  - [Link](https://link.springer.com/chapter/10.1007/978-3-319-61845-6_10)
  - **Objective:** Predict sales of weather-sensitive food products, including bakery items, at a Japanese chain supermarket using meteorological data.
  - **Methods:** Utilizes a deep learning approach combining long short-term memory (LSTM) networks and stacked denoising autoencoders, trained on historical sales data paired with weather variables (e.g., temperature, precipitation). Data is structured as time-series with daily sales records.
  - **Outcomes:** The method outperforms traditional machine learning models by 19.3%, effectively predicting sales for weather-sensitive items like bakery goods, based on a large historical dataset spanning multiple years.
  - **Relation to the Project:** Supports the use of neural networks (specifically LSTM) for bakery sales prediction influenced by weather, aligning with your focus on six product categories. Emphasizes time-series data with weather inputs and suggests substantial historical data improves accuracy, with no mention of pretrained models.

---

## Additional Insights
- **Common Models:** Neural network regression, including feedforward ANNs and LSTM variants, is widely adopted for bakery sales prediction due to its strength in modeling non-linear effects of weather and events. These approaches suit the dynamic nature of time-series data with multiple influencing factors.
- **Training Data Format:** Data is consistently organized as daily time-series tables, featuring columns for sales (numeric), weather variables (e.g., temperature, precipitation as continuous values), and event indicators (categorical flags, e.g., "storm = 1" or "holiday = 1").
- **Data Volume:** Studies leverage datasets spanning multiple years, with Source 1 using 5 years (millions of records) and Source 2 implying a similarly large multi-year scope. Larger datasets enhance predictive accuracy, though the exact volume varies by context.
- **Pretrained Models:** Neither source employs pretrained models; both develop custom neural networks tailored to their specific datasets and objectives, indicating this is a common practice for specialized retail forecasting tasks like bakery sales.

---