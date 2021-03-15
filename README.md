# Assignment1

ref：Andrey Shtrauss. 'Building an Asset Trading Strategy', 2020.
https://www.kaggle.com/shtrausslearning/building-an-asset-trading-strategy

This kaggle passage develop a mechine learning stragtegy for Bitcoin trading. I'm going to implement the same strategy in Shanghai Securities Composite Index.

Use trading signals as target variables to extract feature values from price information and technical indicators for machine learning. Specifically, taking the dual moving average trading strategy as an example, a buy signal is formed when the short-term moving average breaks above the long-term moving average (set to 1), and a sell signal is issued when the short-term moving average falls below the long-term moving average (set to 0), so that the signal variable constitutes a (0-1) categorical variable, which is convenient for us to further use the machine learning model for evaluation. Here, the parameters of the short-term moving average (SMA1) and long-term moving average (SMA2) are set to 10 and 60 respectively. The settings of the two have a certain degree of arbitrariness. The choice of parameters will affect the subsequent results, so ideally it is required Perform parameter optimization to find the optimal value.

![Figure_5](https://user-images.githubusercontent.com/78809297/111138773-4ea8dc80-85bb-11eb-87a1-e921c247b912.png)

![Figure6](https://user-images.githubusercontent.com/78809297/111138790-523c6380-85bb-11eb-8257-6f1a9c3603d0.png)

![Figure_1](https://user-images.githubusercontent.com/78809297/111138807-56688100-85bb-11eb-95b5-482ba17843fc.png)

The linear correlation values between the current features open, high, low, close, volumes and the target variable are very small. Maybe they are not ideal predictive feature variables, so feature construction and selection are needed below.

To facilitate analysis, the following introduces a feature matrix with technical indicators as features, which specifically include the following technical indicators:

Moving average: Moving average indicates the trend of price movement by reducing noise.
Stochastic Oscillator %K and %D: Stochastic Oscillator is a momentum indicator that compares the closing price of a particular security with the price range within a certain period of time. %K and %D are slow and fast indicators respectively.
Relative Strength Index (RSI): A momentum indicator that measures the magnitude of recent price changes to assess the overbought or oversold conditions of stocks or other assets.
Rate of Change (ROC): Momentum oscillator, which measures the percentage change between the current price and the past price in n periods. The higher the ROC value, the more likely to be overbought, and the lower the ROC value, the more likely it is to be oversold.
Momentum (MOM): The speed at which the price or volume of a security accelerates; the speed at which prices change.

![Figure_2](https://user-images.githubusercontent.com/78809297/111138834-5cf6f880-85bb-11eb-8376-b21c994d0d81.png)

![Figure_3](https://user-images.githubusercontent.com/78809297/111137631-fd4c1d80-85b9-11eb-9f69-dd6a514e08aa.png)

![Figure_4](https://user-images.githubusercontent.com/78809297/111137645-01783b00-85ba-11eb-9c45-b88197ad4d87.png)


