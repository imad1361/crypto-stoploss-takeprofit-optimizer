import pandas as pd
import ccxt
from backtesting import Backtest, Strategy
import talib
import time
import numpy as np

# Initialize the Binance exchange
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# Define the ticker for Bitcoin perpetual futures
TICKER = 'BTC/USDT'

# Define the period for the last month
end_date = pd.Timestamp.now()  # Current date and time
start_date = end_date - pd.DateOffset(months=1)

# Function to fetch historical OHLCV data in chunks
def fetch_historical_data(ticker, since, timeframe='5m'):
    all_data = []
    while since < int(end_date.timestamp() * 1000):
        data = exchange.fetch_ohlcv(ticker, timeframe=timeframe, since=since)
        if not data:
            break
        all_data += data
        since = data[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)
    return all_data

# Fetching 5-minute historical data
data = fetch_historical_data(TICKER, int(start_date.timestamp() * 1000))

# Convert to DataFrame
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print(df.head())
print(f"Total rows fetched: {len(df)}")

class RsiStrategy(Strategy):
    rsi_period = 14
    upper_bound = 70  # Short when RSI > 70
    lower_bound = 28   # Long when RSI < 28

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)

    def next(self):
        price = self.data.Close[-1]

        if not self.position:
            # Long entry condition
            if self.rsi[-1] < self.lower_bound:
                sl = price * (1 - self.stop_loss_pct)
                tp = price * (1 + self.take_profit_pct)
                self.buy(sl=sl, tp=tp)

            # Short entry condition
            elif self.rsi[-1] > self.upper_bound:
                sl = price * (1 + self.stop_loss_pct)
                tp = price * (1 - self.take_profit_pct)
                self.sell(sl=sl, tp=tp)

# Define ranges for stop loss and take profit percentages
stop_loss_range = np.arange(0.002, 0.1, 0.001)  # From 0.2% to 2% (exclusive)
take_profit_range = np.arange(0.003, 0.1, 0.001)  # From 0.2% to 3% (exclusive)

best_return = -np.inf
best_params = (None, None)

results = []

# Iterate through stop loss and take profit combinations
for sl in stop_loss_range:
    for tp in take_profit_range:
        # Create a new instance of the strategy with current parameters
        class CustomRsiStrategy(RsiStrategy):
            stop_loss_pct = sl
            take_profit_pct = tp

        # Run backtest with current parameters
        bt = Backtest(df, CustomRsiStrategy, cash=1000000, commission=0.002)
        stats = bt.run()
        
        total_return = stats['Return [%]']
        results.append((sl * 100, tp * 100, total_return))

        # Check for best parameters based on total return
        if total_return > best_return:
            best_return = total_return
            best_params = (sl * 100, tp * 100)

# Print best parameters and their return
print(f"Best Stop Loss: {best_params[0]:.2f}%")
print(f"Best Take Profit: {best_params[1]:.2f}%")
print(f"Best Total Return: {best_return:.2f}%")

# Convert results to DataFrame for plotting
results_df = pd.DataFrame(results, columns=['Stop Loss (%)', 'Take Profit (%)', 'Total Return (%)'])

# Plotting results using matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(results_df['Stop Loss (%)'], results_df['Take Profit (%)'], c=results_df['Total Return (%)'], cmap='viridis')
plt.colorbar(label='Total Return (%)')
plt.title('Total Return based on Stop Loss and Take Profit')
plt.xlabel('Stop Loss (%)')
plt.ylabel('Take Profit (%)')
plt.grid()
plt.show()
