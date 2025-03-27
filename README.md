##Problem Definition:

The goal of this project is to predict interest rates for the first few months of 2025 using historical FedFunds interest rate data. Interest rates are a critical economic indicator that influences borrowing costs, investment decisions, and broader economic activity. Predicting their future trajectory is difficult, due to the many real-world factors that affect economic stability, but doing so helps financial institutions, policymakers, and investors make informed decisions about future time periods.

##Assumptions:

Stationarity: We assume that historical trends in interest rates will continue in the short term. This is a critical assumption because it implies that the statistical properties (mean, variance) of the time series do not change over time.

Normal Distribution: Interest rates can be treated as a random variable following a normal distribution with some mean and variance. However, this assumption could be weaker in times of economic crises or rapid changes.

Economic Indicators: Interest rates are influenced by macroeconomic factors such as inflation, job reports, and central bank policies. 

Lagged Relationships: Past values of interest rates are predictive of future values and can be used mathematically to do so.

##What Are We Predicting?

Target Variable: Monthly interest rates for the initial months of 2025.
Time Horizon: Short-term predictions (1–5 months).

What Does the Interest Rate Mean?

• The interest rate is the cost of borrowing money, often represented as a percentage of the loan amount. They influence everything from consumer loans to business investments, impacting economic growth.
• However, in this project, we focus on the federal funds rate, which reflects the broader economic environment, rather than specific interest rates in certain areas of the economy.


##Data Used for Prediction:

The data we used for this project was taken from: https://fred.stlouisfed.org/series/FEDFUNDS

##What does this data mean?

The data is of the Federal Fund Rates from 1954-2024 for each month. The shape of the obtained data is (845,2). We have only 2 features, being the Date and Federal Fund Rate. The date column represents the first day of every month from July 1954 to November 2024. The FEDFUNDS column provides information about the interest rate the Federal Bank has decided for the respective month. 

As we are utilizing historical data for the upcoming month's predictions, it is highly vulnerable to current market situations and trends. Below are some of the challenges we might face with the data:

Exogenous Variables:
Factors like unexpected geopolitical events or pandemics can disrupt patterns, making forecasts less reliable.
We can observe from the data from 2020–2021 (0.08–0.09%) that there were extremely low federal fund rates, which are a result of the economic disruptions caused by the COVID-19 pandemic.
Assumption of Trends:
Interest rate trends are often influenced by central bank decisions, which may not always follow historical patterns. After the pandemic, we could see the decision making of central government as economies reopened. Central banks shifted focus from stimulus to managing overheating economies.
Result: Gradual interest rate hikes beginning in early 2022.

We can also observe how the period of 2013–2014 shows consistently low rates, ranging between 0.07% and 0.09%. This trend aligns with the aftermath of the 2008 financial crisis, during which central banks around the world, particularly the U.S. Federal Reserve, maintained near-zero interest rates for an extended period to support economic recovery.


##Model Selection:
To gain an understanding of the limited data that we have, we tried implementing ARIMA, SARIMA, and LSTM Models, testing out both Neural Networks and Time Series Model implementation.
What Features of the Data Are You Trying to Establish Relationships Between?
The key feature we are trying to establish relationships between is the interest rate (FEDFUNDS), which is our target variable, with the date (timestamp). 
Interest Rate (FEDFUNDS): This is the dependent variable, or the target, we want to forecast. It represents the Federal Funds Effective Rate, a critical interest rate that banks use when lending to each other overnight. The rate influences borrowing costs across the economy, affecting everything from mortgage rates to business loans.
Date (Timestamp): This is the independent variable that provides the temporal context for our interest rate data. We are trying to understand how the interest rate changes over time, potentially driven by economic trends, policy decisions, and macroeconomic cycles. The relationship between the interest rate and time could involve trends (e.g., increasing or decreasing over the years) and seasonality (e.g., periodic adjustments made based on fiscal years, or economic cycles).
In simpler models like SARIMA, the date feature helps the model capture seasonality (such as quarterly adjustments or annual cycles) and trends (long-term shifts in interest rate policies or economic conditions).
More advanced models like LSTM also tries to capture complex temporal dependencies, where the interest rate at a given time could be influenced by multiple past values of the rate itself.
In summary, the primary relationship we're exploring is between the date (time) and the interest rate (target variable), with the goal of understanding how past interest rates affect future rates, considering potential trends and seasonal patterns.

##Model Architecture:
There are several models used in this project. They are:
1.ARIMA (AutoRegressive Integrated Moving Average)
ARIMA is a widely used time series forecasting model that combines three components: AutoRegressive (AR), Integrated (I), and Moving Average (MA). Here’s a breakdown of the mathematical representation of ARIMA, typically represented as ARIMA(p, d, q):
Mathematical Representation:
 

Where:
Yt is the time series value at time t (interest rate at time t in the case of the Fed Funds Rate).
μ is the constant or mean of the series, which could be the overall average value of the series.
p is the order of the AutoRegressive (AR) part. It indicates how many previous time steps (lags) should be used to predict the current value. It captures the linear dependency between an observation and its previous p observations.
d is the number of times the series is differenced to make it stationary. 
q is the order of the Moving Average (MA) part. It refers to how many previous error terms (residuals from the predictions) should be included in the model.
ϵt is the white noise (error term) at time t, which is assumed to be a random error with zero mean and constant variance.
Summary of ARIMA Terms:
•	AR: Auto-regressive terms ϕ1,ϕ2,…,ϕp
•	I: Differencing term (degree of differencing, d).
•	MA: Moving average terms θ1,θ2,…,θq
2. SARIMA (Seasonal ARIMA)
SARIMA is an extension of ARIMA that includes seasonal components to handle seasonality in time series data. The mathematical representation of SARIMA is SARIMA(p, d, q)(P, D, Q)[s] where:
p, d, q: These are the same as in ARIMA (non-seasonal parameters).
P, D, Q: Seasonal components of the ARIMA model. These are each similar to their non-seasonal counterpart, instead using past seasonal terms instead of past terms directly.
s: The length of the seasonal cycle (e.g., 12 for monthly data with annual seasonality).

3. LSTM (Hyperparameters tuned)
Input Layer: (12,8) 12 time steps, 8 features
1st Layer: LSTM (64) - Dropout (10%)
2nd Layer: Dense (64, ReLu Activation [to reduce complexity])
Output: 1 unit (Linear Activation)
Architecture and Relationship Between Data and Prediction
In both Neural Networks and Time Series models, the architecture is designed to model the relationships between the features and the prediction (future interest rate).
Neural Networks (e.g., LSTM):
LSTM networks capture the sequential dependencies in the Fed Funds Rate data. The LSTM architecture allows the model to "remember" past interest rate values for longer periods, which is essential when forecasting rates based on historical data.
•	Example: If the interest rate increased over the last few months due to a particular economic event (e.g., inflation concerns), the LSTM can learn that pattern and use it to predict whether the rate will continue to rise in the near future.
Hidden Layers: As the input data (past interest rates) moves through the hidden layers, the network learns complex patterns and relationships between different time periods, capturing trends and seasonality (if present).
•	Example: Hidden layers in the neural network might learn that interest rates typically rise during certain months of the year or after major policy announcements.
ARIMA, SARIMA:
ARIMA captures the relationship between the current value and previous time steps (lags). It assumes that past values of the Federal Funds Rate influence future values. For example, if the rate has been rising recently, it may continue to rise.
Seasonality (SARIMA): If the Fed Funds Rate exhibits seasonal patterns (e.g., changes based on fiscal years or regular policy reviews), SARIMA models can account for that seasonality and adjust predictions for months or quarters where seasonal effects are strong.
Evaluation Metrics:
Our task involves regression, predicting continuous values (Federal Funds Effective Rates). The selected metrics—MSE, RMSE, and MAE—are standard and widely accepted for evaluating regression models due to their ability to quantify prediction accuracy effectively.
Results and Interpretation:

Models	MSE	RMSE
ARIMA	0.2089	0.4570
SARIMA	0.1926	0.4388
Exponential Smoothing	0.2332	0.4829
LSTM	0.0495	0.2225


ARIMA: This model was one of the worse performing models. This is most likely due to its simplicity compared to the other models, so it failed to represent the complexity of the data.
SARIMA: This model performed better than the ARIMA, which is due to it capturing any hidden seasonal trends in the data, allowing it to pick up on some of the more complex trends, but it still wasn’t enough to have a massive improvement over the others.
Exponential Smoothing: This model performed the worst out of the ones tested, which is surprising, given its ability to weigh recent time periods more than past ones, which should have led it to performing better than the others.
LSTM: The model's predictions deviate from the actual values by approximately 0.2703 percentage points on average. This value is directly interpretable in the context of interest rate percentages. Due to the very low RMSE value, we can conclude that LSTM is performing really well with this data, and is the best model out of the ones tested.

Out of all of our models, LSTM performed the best. LSTM can learn complex hidden patterns from the data, they can assign weights to different lags along with the pass gates and forgot gates, making it much more powerful than the rest. ARIMA, SARIMA, and Exponential Smoothing can be effective in predicting simple data models, not as much for data with complex trends.
![image](https://github.com/user-attachments/assets/46827700-8680-40ee-857d-152e90f7a4e0)
