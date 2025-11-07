# Stock Price Prediction using Machine Learning

## Description

This project explores the feasibility of predicting stock price movements using machine learning. The primary goal is to determine if historical price data and technical indicators contain sufficient information to forecast the direction of the next price movement with a meaningful level of accuracy.

The approach is structured as a complete end-to-end data science project, encompassing:

1.  **Data Gathering and Processing**: Financial data (candlestick data) is autonomously collected from multiple sources, including Yahoo Finance and Alpaca API, to build a comprehensive and up-to-date dataset. A robust data processing pipeline cleans the data and engineers a wide array of features, including various technical indicators like RSI, MACD, Bollinger Bands, and ATR, as well as time-based cyclical features.

2.  **Model Training and Selection**: The project adopts a binary classification approach. The model predicts whether the closing price of the next candlestick will be higher (1) or lower (0) than the current one. An extensive grid search is performed using a machine learning pipeline that includes preprocessing, optional dimensionality reduction, and various classification algorithms (e.g., Logistic Regression, Support Vector Machines, Random Forest, XGBoost). The models are rigorously evaluated using cross-validation and the Matthews Correlation Coefficient (MCC) to handle potential class imbalance.

3.  **Results Analysis and Trading Simulation**: The performance of the best models is analyzed in detail. To assess the practical application of the model's predictions, a high-leverage trading strategy is simulated. The strategy's performance is benchmarked against a simple "Buy-and-Hold" approach and evaluated against the strict criteria of the FTMO Challenge, a set of rules used by proprietary trading firms to assess trader performance, focusing on risk management aspects like daily loss limits and maximum drawdown.

This study provides a thorough analysis of building a stock prediction model from scratch, highlighting both the potential and the inherent limitations and risks of relying on purely technical analysis for financial forecasting.

**Disclaimer**: The information and results presented in this project are for educational and research purposes only and do not constitute financial advice. Trading in financial markets involves significant risk, and any investment decisions should be made with professional guidance.

## Installation

To run this project, you need to have Python installed. You can install the required libraries using pip:

```bash
pip install --upgrade pip
pip install yfinance==0.2.61
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install matplotlib==3.9.2
pip install mlxtend==0.23.3
pip install imblearn==0.0
pip install scikit-learn==1.5.2
pip install missingno==0.5.2
pip install alpaca_trade_api==3.2.0
pip install xgboost
pip install MetaTrader5
```

## Usage

The project is contained within a single Jupyter Notebook (`.ipynb` file). To use it, you can follow the steps outlined below:

### 1. Data Gathering

The first part of the notebook focuses on data acquisition. The `download_data_history` function allows you to download historical stock data. You can configure several parameters:
- `ticker`: The stock symbol you want to analyze (e.g., `^GSPC` for S&P 500, `^FTSE` for FTSE 100).
- `candle_time`: The timeframe for each candlestick (e.g., `1d` for daily, `1h` for hourly, `5m` for 5 minutes).
- `days_delta`: The number of past days of data to download.
- `download_option`: The source of the data (`yf` for Yahoo Finance, `alpaca`, `mixed`, `crypto`, `mt5`).

Example:
```python
input_file_path = 'assets/data_history.csv'
download_data_history(candle_time="1d", path=input_file_path, download_option="yf", ticker="^FTSE", days_delta=10000)
```

### 2. Feature Engineering and Preprocessing

Once the data is downloaded, the notebook processes it to create features for the machine learning model. The `extract_advanced_features` function calculates a wide range of technical indicators and creates lagged features from previous candlesticks. The `add_classification_label` function generates the binary target variable. This entire process is executed in the "Data Preprocessing and Saving" section.

### 3. Model Training and Evaluation

The second part of the notebook is dedicated to creating and training the machine learning model.
- The data is split into training and testing sets.
- A `Pipeline` is constructed, which includes data scaling, optional dimensionality reduction (e.g., PCA), and a classifier.
- A `RandomizedSearchCV` is used to perform an extensive grid search over a combination of different dimensionality reduction techniques, classifiers, and their respective hyperparameters.
- The best-performing models are identified based on the Matthews Correlation Coefficient (MCC) score.

### 4. Results Analysis and Trading Simulation

The third part analyzes the results of the trained models.
- The classification reports and confusion matrices for the top models are displayed.
- The `process_and_analyze` function visualizes the model's predictions and calculates its accuracy.
- The `evaluate_trading_strategy_and_verify` function simulates a trading strategy based on the model's predictions. This function outputs:
    - A plot comparing the strategy's capital growth against a Buy-and-Hold benchmark.
    - A bar chart showing the distribution of incorrect predictions per month.
    - Key performance metrics like final capital, maximum drawdown, and maximum consecutive losses.
    - A verification of the strategy against FTMO Challenge criteria.

You can adjust the parameters of the simulation, such as `initial_capital`, `leverage`, and `prediction_probability` threshold to test different scenarios.

Example:
```python
df_results = pd.read_csv("output_model_1.csv")
evaluate_trading_strategy_and_verify(df_results, prediction_probability=0.6)```

## Results

The experimentation revealed several key insights:

-   **Best Performing Models**: Logistic Regression and Linear Support Vector Classifier (SVC) consistently emerged as the top-performing algorithms.
-   **Impact of Timeframe**: The models achieved higher accuracy on longer timeframes. Daily candles yielded an accuracy of around 60%, while hourly candles produced approximately 55%. Despite the lower accuracy, the hourly timeframe allows for a higher frequency of trades, which can be advantageous.
-   **Asset Selection**: The models performed significantly better on broad market indices (like the S&P 500) compared to individual stocks or currency pairs. This is likely because indices are less susceptible to the high volatility caused by company-specific news, which this model does not account for.
-   **Trading Simulation**: A simulated trading strategy using 100x leverage on the S&P 500 with hourly candles over a 6-month test period showed promising results:
    -   **Capital Growth**: An initial capital of $10,000 grew to approximately $15,000–$16,000 (a 50-60% increase).
    -   **Risk Metrics**: The strategy maintained a maximum drawdown of around 9-11% and a maximum daily loss percentage of about 4%, successfully passing the simulated FTMO challenge criteria for risk management.
-   **Probability Filter**: Using the model's prediction probability as a filter for trades proved effective. By setting a threshold (e.g., only trading when the prediction probability is > 60%), the number of trades decreases, but the overall accuracy of the executed trades increases significantly.

## Future Improvements

The project lays a solid foundation, but several areas can be explored for enhancement:

1.  **Generalization**: Test the model on a wider variety of stocks and market conditions to build a more robust and generalized predictor.
2.  **Portfolio Diversification**: Construct a portfolio of multiple assets to diversify risk. Incorporate transaction costs into the simulation for a more realistic evaluation of profitability.
3.  **Advanced Risk Management**: Implement dynamic stop-loss and take-profit strategies based on market volatility (e.g., using the ATR indicator) rather than fixed percentages.
4.  **Unified Data Source**: Utilize a single, reliable data provider with extensive historical data and live access to ensure data consistency and model stability.
5.  **Non-Technical Features**: Integrate sentiment analysis from financial news, social media, and earnings reports using Natural Language Processing (NLP) to capture market sentiment, which is a significant driver of price movements, especially for individual stocks.

## References

-   Chen, Y., & Jiang, S. (2015). *Stock market forecasting using machine learning algorithms*. Stanford University, CS229 Project Report.
-   Dai, X., & Zhang, Y. (2013). *Machine learning in stock price trend forecasting*. Stanford University, CS229 Project Report.
-   Shen, J., Jiang, S., & Zhang, Y. (2012). *Stock market forecasting using machine learning algorithms*. Stanford University, CS229 Project Report.
-   Scorpionhiccup. (n.d.). *Stock price prediction*. GitHub repository.
-   Scikit-learn developers. (n.d.). *Scikit-learn: Machine learning in Python*.
-   StockCharts.com. (n.d.). *Chart school*.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

This license allows you to:
-   **Share**: copy and redistribute the material in any medium or format.
-   **Adapt**: remix, transform, and build upon the material.

Under the following conditions:
-   **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
-   **NonCommercial**: You may not use the material for commercial purposes.

For more information, visit the [full license](https://creativecommons.org/licenses/by-nc/4.0/).