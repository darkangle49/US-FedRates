# Interest Rate Prediction Project

## Problem Definition

The goal of this project is to predict interest rates for the first few months of 2025 using historical FedFunds interest rate data. Interest rates are a critical economic indicator that influences borrowing costs, investment decisions, and broader economic activity. Predicting their future trajectory is difficult, due to the many real-world factors that affect economic stability, but doing so helps financial institutions, policymakers, and investors make informed decisions about future time periods.

## Assumptions

- **Stationarity**: We assume that historical trends in interest rates will continue in the short term. This is a critical assumption because it implies that the statistical properties (mean, variance) of the time series do not change over time.
- **Normal Distribution**: Interest rates can be treated as a random variable following a normal distribution with some mean and variance. However, this assumption could be weaker in times of economic crises or rapid changes.
- **Economic Indicators**: Interest rates are influenced by macroeconomic factors such as inflation, job reports, and central bank policies.
- **Lagged Relationships**: Past values of interest rates are predictive of future values and can be used mathematically to do so.

## What Are We Predicting?

- **Target Variable**: Monthly interest rates for the initial months of 2025.
- **Time Horizon**: Short-term predictions (1–5 months).

## What Does the Interest Rate Mean?

- The interest rate is the cost of borrowing money, often represented as a percentage of the loan amount. They influence everything from consumer loans to business investments, impacting economic growth.
- In this project, we focus on the **federal funds rate**, which reflects the broader economic environment, rather than specific interest rates in certain areas of the economy.

## Data Used for Prediction

The data we used for this project was taken from:  
[https://fred.stlouisfed.org/series/FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS)

### What does this data mean?

- The data is of the Federal Fund Rates from 1954-2024 for each month. The shape of the obtained data is (845, 2). We have only 2 features, being the **Date** and **Federal Fund Rate**.
- The **Date** column represents the first day of every month from July 1954 to November 2024.
- The **FEDFUNDS** column provides information about the interest rate the Federal Bank has decided for the respective month.

### Challenges

- **Exogenous Variables**: Factors like unexpected geopolitical events or pandemics can disrupt patterns, making forecasts less reliable.
    - Example: Extremely low federal fund rates from 2020–2021 were a result of the economic disruptions caused by the COVID-19 pandemic.
- **Assumption of Trends**: Interest rate trends are often influenced by central bank decisions, which may not always follow historical patterns.
    - Example: Gradual interest rate hikes beginning in early 2022.

## Model Selection

We experimented with different models to understand the limited data and predict interest rates:
- **ARIMA (AutoRegressive Integrated Moving Average)**
- **SARIMA (Seasonal ARIMA)**
- **LSTM (Long Short-Term Memory)**

### Features We Are Trying to Establish Relationships Between

The key feature we are trying to establish relationships between is the **interest rate (FEDFUNDS)**, which is our target variable, with the **date (timestamp)**.

- **Interest Rate (FEDFUNDS)**: This is the dependent variable, or the target, we want to forecast.
- **Date (Timestamp)**: This is the independent variable that provides the temporal context for our interest rate data.

### Model Architecture

#### 1. ARIMA (AutoRegressive Integrated Moving Average)

- ARIMA combines three components: AutoRegressive (AR), Integrated (I), and Moving Average (MA).
- **Mathematical Representation**:
  $$ Y_t = \mu + \sum_{i=1}^{p} \phi_i Y_{t-i} + \sum_{j=1}^{q} \theta_j \epsilon_{t-j} + \epsilon_t $$

Where:
- \( Y_t \) is the time series value at time \( t \) (interest rate at time \( t \)).
- \( \mu \) is the constant or mean of the series.
- \( p \) is the order of the AutoRegressive (AR) part.
- \( d \) is the number of times the series is differenced to make it stationary.
- \( q \) is the order of the Moving Average (MA) part.
- \( \epsilon_t \) is the white noise (error term).

#### 2. SARIMA (Seasonal ARIMA)

- SARIMA extends ARIMA to include seasonal components.
- **Mathematical Representation**:
  $$ \text{SARIMA}(p, d, q)(P, D, Q)[s] $$

Where:
- \( p, d, q \) are the non-seasonal components.
- \( P, D, Q \) are the seasonal components.
- \( s \) is the length of the seasonal cycle.

#### 3. LSTM (Long Short-Term Memory)

- Input Layer: (12, 8) — 12 time steps, 8 features
- 1st Layer: LSTM (64) - Dropout (10%)
- 2nd Layer: Dense (64, ReLU Activation)
- Output: 1 unit (Linear Activation)

LSTM models capture sequential dependencies in the data, allowing them to "remember" past interest rate values and use that information to predict future rates.

## Evaluation Metrics

We used the following metrics to evaluate our models:
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

### Results and Interpretation

| Model                | MSE    | RMSE   |
|----------------------|--------|--------|
| ARIMA                | 0.2089 | 0.4570 |
| SARIMA               | 0.1926 | 0.4388 |
| Exponential Smoothing | 0.2332 | 0.4829 |
| LSTM                 | 0.0495 | 0.2225 |

- **ARIMA**: This model was one of the worst performing models. This is likely because its simplicity didn't capture the complexity of the data.
- **SARIMA**: This model performed better than ARIMA because it captured hidden seasonal trends but still didn't offer significant improvement.
- **Exponential Smoothing**: Surprisingly, this model performed the worst, even though it should weigh recent periods more heavily.
- **LSTM**: The LSTM model performed the best, with an RMSE value of 0.2225, meaning it deviated from actual values by approximately 0.2703 percentage points on average. The LSTM can learn complex patterns and assign weights to different lags, making it superior to other models.

### Conclusion

Out of all the models tested, **LSTM** performed the best. It can learn complex hidden patterns from the data and handle sequential dependencies better than traditional models like ARIMA, SARIMA, and Exponential Smoothing. These traditional models are effective for simple data patterns but struggle with complex trends like those in the Federal Funds Rate data.

## Future Work

Future improvements could include:
- Integrating exogenous variables like inflation, job reports, and central bank policies to improve prediction accuracy.
- Experimenting with more advanced neural network architectures like GRU (Gated Recurrent Units) or Transformer-based models.
- Exploring a more comprehensive evaluation approach by considering the impact of geopolitical events and market shifts on interest rates.

