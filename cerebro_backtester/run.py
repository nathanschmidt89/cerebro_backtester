import pandas as pd
from backtest import Backtest

# Example historical data
data = pd.DataFrame({
    'Close': [100, 102, 101, 98, 96],
    'High': [101, 103, 102, 100, 97],
    'Low': [99, 100, 100, 95, 94],
    'buy_signal': [1, 0, 0, 1, 0],
    'sell_signal': [0, 0, 1, 0, 1]
})

# Configure and run the backtest
backtest = Backtest(
    info=data,
    side='long',
    SL_type='atr',
    SL=2,
    SL_spike_out=True,
    TP=3,
    TS=True
)

# Run backtest
backtest.run()

# Print results
print(backtest.info[['equity_curve', 'drawdown', 'trade_return_hist']])


