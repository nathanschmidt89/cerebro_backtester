
# Backtest Framework

This Python backtesting framework allows you to evaluate trading strategies on historical market data. It supports stop-loss (SL), take-profit (TP), trailing stop (TS), and various risk management features. You can backtest both long and short strategies with configurable settings for transaction costs and balance management.

Features
Backtest long and short trading strategies
Configurable stop-loss (SL) types: ATR-based, standard deviation (SD)-based, and percentage-based
Take-profit and trailing stop functionality
Risk management through percentage-based position sizing
Transaction costs handling
Equity curve and drawdown tracking
Detailed trade history, including entry and exit points, stop-loss updates, and trade returns


## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>


## Installation

Install cerebro_backtester with pip

```python
  pip install cerebro-backtester
```


## Usage/Examples

### Example 1: Simple Long Strategy with ATR-based Stop Loss

```python
import pandas as pd
from backtest import Backtest

# Example historical data
data = pd.DataFrame({
    'Close': [100, 102, 101, 104, 106],
    'High': [101, 103, 102, 105, 107],
    'Low': [99, 100, 100, 103, 105],
    'buy_signal': [1, 0, 0, 1, 0],
    'sell_signal': [0, 0, 0, 0, 1]
})

# Configure and run the backtest
backtest = Backtest(
    info=data,
    side='long',
    SL_type='atr',
    SL=2,  # 2x ATR for stop-loss
    SL_spike_out=True,
    TP=3,  # 3x ATR for take-profit
    TS=True  # Enable trailing stop
)
backtest.run()

# Analyze results
print(backtest.info[['equity_curve', 'drawdown', 'trade_return_hist']])
```

### Example 2: Short Strategy with Standard Deviation-based Stop Loss

```python
import pandas as pd
from backtest import Backtest

# Example historical data
data = pd.DataFrame({
    'Close': [100, 102, 101, 98, 96],
    'High': [101, 103, 102, 100, 97],
    'Low': [99, 100, 100, 95, 94],
    'buy_signal': [0, 0, 0, 1, 0],
    'sell_signal': [1, 0, 1, 0, 0]
})

# Configure and run the backtest
backtest = Backtest(
    info=data,
    side='short',
    SL_type='sd',
    SL=1,  # 1x standard deviation for stop-loss
    SL_spike_out=False,
    TP=2,  # 2x standard deviation for take-profit
    TS=False  # Disable trailing stop
)
backtest.run()

# Analyze results
print(backtest.info[['equity_curve', 'drawdown', 'trade_return_hist']])
```


## Authors

Nathan Schmidt is a  programmer with years of experience in software development. Specializing in innovative programming solutions, Nathan has a commitment to open-source development and community collaboration. Known for his passion for clean code and efficiency, Nathan continues to contribute to the field of software engineering with a focus on impactful, real-world applications.


## Support

For support, email nathan.schmidt.ns89@gmail.com or raise a Github issue.