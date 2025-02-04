# Crypto Stop-Loss & Take-Profit Optimizer

## Finding the Best Stop-Loss & Take-Profit Levels for BTC Perpetuals/Crypto

### Overview
This project is designed to backtest various stop-loss and take-profit percentages for BTC/USDT perpetual futures on Binance. Using historical data and an RSI-based strategy, it determines the best risk management levels for optimal profitability.

### Features
- Fetches historical 5-minute OHLCV data from Binance
- Uses Relative Strength Index (RSI) for trading signals
- Backtests multiple stop-loss and take-profit combinations
- Finds optimal SL & TP percentages for highest return
- Plots results using Matplotlib for better visualization

### Installation
To run this script, install the required dependencies using:
```bash
pip install pandas ccxt backtesting numpy talib matplotlib
```

### How It Works

#### 1. Fetch Historical Data
- Uses `ccxt` to get 5-minute OHLCV (Open, High, Low, Close, Volume) data for BTC/USDT from Binance.
- Fetches data for the last month by making API calls in chunks (to avoid rate limits).
- Converts the data into a Pandas DataFrame for analysis.

#### 2. Implement Trading Strategy (RSI-based)
- Uses a custom RSI-based strategy:
  - Long (Buy) when RSI < 28
  - Short (Sell) when RSI > 70
- Applies stop-loss and take-profit to control risk.
- Backtests different SL & TP combinations to find the most profitable one.

#### 3. Run Backtesting for Multiple SL & TP Values
- Tries different stop-loss (%) values from 0.2% to 10%
- Tries different take-profit (%) values from 0.3% to 10%
- Runs backtests using `Backtesting.py` for every SL-TP pair
- Finds the combination that gives the highest total return

#### 4. Visualize Results
- Saves all SL-TP return values into a Pandas DataFrame
- Uses `matplotlib` to plot a scatter plot of stop-loss vs. take-profit vs. return
- Helps traders see which SL-TP combinations maximize profits

### Code Breakdown

#### 1. Import Dependencies
```python
import pandas as pd
import ccxt
from backtesting import Backtest, Strategy
import talib
import time
import numpy as np
import matplotlib.pyplot as plt
```

#### 2. Fetch Historical Data
```python
def fetch_historical_data(ticker, since, timeframe='5m'):
    all_data = []
    while since < int(end_date.timestamp() * 1000):
        data = exchange.fetch_ohlcv(ticker, timeframe=timeframe, since=since)
        if not data:
            break
        all_data += data
        since = data[-1][0] + 1  # Move to the next timestamp
        time.sleep(exchange.rateLimit / 1000)  # Avoid rate limit
    return all_data
```

#### 3. Create RSI Strategy for Trading
```python
class RsiStrategy(Strategy):
    rsi_period = 14
    upper_bound = 70
    lower_bound = 28

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)

    def next(self):
        price = self.data.Close[-1]

        if not self.position:
            if self.rsi[-1] < self.lower_bound:
                sl = price * (1 - self.stop_loss_pct)
                tp = price * (1 + self.take_profit_pct)
                self.buy(sl=sl, tp=tp)
            elif self.rsi[-1] > self.upper_bound:
                sl = price * (1 + self.stop_loss_pct)
                tp = price * (1 - self.take_profit_pct)
                self.sell(sl=sl, tp=tp)
```

#### 4. Backtest Different SL & TP Values
```python
stop_loss_range = np.arange(0.002, 0.1, 0.001)
take_profit_range = np.arange(0.003, 0.1, 0.001)

best_return = -np.inf
best_params = (None, None)
results = []

for sl in stop_loss_range:
    for tp in take_profit_range:
        class CustomRsiStrategy(RsiStrategy):
            stop_loss_pct = sl
            take_profit_pct = tp

        bt = Backtest(df, CustomRsiStrategy, cash=1000000, commission=0.002)
        stats = bt.run()
        
        total_return = stats['Return [%]']
        results.append((sl * 100, tp * 100, total_return))

        if total_return > best_return:
            best_return = total_return
            best_params = (sl * 100, tp * 100)
```

#### 5. Plot the Results
```python
results_df = pd.DataFrame(results, columns=['Stop Loss (%)', 'Take Profit (%)', 'Total Return (%)'])

plt.figure(figsize=(12, 6))
plt.scatter(results_df['Stop Loss (%)'], results_df['Take Profit (%)'], c=results_df['Total Return (%)'], cmap='viridis')
plt.colorbar(label='Total Return (%)')
plt.title('Total Return based on Stop Loss and Take Profit')
plt.xlabel('Stop Loss (%)')
plt.ylabel('Take Profit (%)')
plt.grid()
plt.show()
```

### Results Example
```
Total rows fetched: 26430
Best Stop Loss: 4.20%
Best Take Profit: 6.30%
Best Total Return: 30.32%

### Result Visualization

![Strategy Visualization](https://github.com/user-attachments/assets/1aead523-2758-48a5-9fb8-fd848ad25908)

This shows the best stop loss and take profit for BTC perpetuals for 1 month.  
You can change the asset from BTC to your desired one, modify the time frame,  
and adjust the RSI upper and lower limits to experiment with different values and optimize the strategy.

This means setting SL at 4.2% and TP at 6.3% would have yielded the highest return based on historical data.
### Disclaimer

This project is intended for educational and informational purposes only. Trading cryptocurrencies involves significant risk, and past performance is not indicative of future results. The author of this repository is not responsible for any financial losses incurred while using this code. Use at your own risk.

### License
This project is open-source under the MIT License.

### How to Contribute
- Fork the repo
- Create a new branch
- Submit a pull request

### Next Steps
- 🚀 Extend the strategy to support multiple timeframes
- 📈 Add more technical indicators like EMA, MACD
- 🔗 Integrate real-time trading with Binance API


