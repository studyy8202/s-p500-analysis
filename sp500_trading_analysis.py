import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load S&P 500 data
# Replace 'data.csv' with the path to your S&P 500 data file
# The data should have columns like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Volatility Analysis
# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# Calculate rolling volatility (standard deviation of returns)
data['Volatility'] = data['Returns'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility

# Momentum Indicators
# Calculate RSI
def compute_rsi(data, window=14):
    delta = data['Close'].diff()  
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data)

# Calculate MACD
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26

# Volume Analysis
# Calculate average volume
data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()

# Gap Trading
# Identify gaps
data['Gap'] = data['Open'] - data['Close'].shift(1)

# Risk Metrics
# Calculate Sharpe Ratio
sharpe_ratio = np.sqrt(252) * (data['Returns'].mean() / data['Returns'].std())

# Calculate Maximum Drawdown
cumulative_returns = (1 + data['Returns']).cumprod()
rolling_max = cumulative_returns.cummax()
max_drawdown = (rolling_max - cumulative_returns) / rolling_max
max_drawdown = max_drawdown.max()

# Trading Recommendations
# Basic strategy based on RSI and MACD
conditions = [
    (data['RSI'] < 30) & (data['MACD'] > 0),  # Buy signal
    (data['RSI'] > 70) & (data['MACD'] < 0)   # Sell signal
]
choices = ['Buy', 'Sell']
data['Signal'] = np.select(conditions, choices, default='Hold')

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.title('S&P 500 Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Display analysis results
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Max Drawdown: {max_drawdown}')
print(data[['Close', 'RSI', 'MACD', 'Signal']].tail())
