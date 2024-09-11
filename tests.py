# Test 1: Check if Equity Curve is Generated
def test_equity_curve():
    data = pd.DataFrame({
        'Close': [100, 102, 104],
        'High': [101, 103, 105],
        'Low': [99, 101, 103],
        'buy_signal': [1, 0, 0],
        'sell_signal': [0, 0, 1]
    })
    backtest = Backtest(data, 'long', 'atr', SL=2, SL_spike_out=True, TP=3, TS=False)
    backtest.run()
    assert 'equity_curve' in backtest.info.columns, "Equity curve not generated"

# Test 2: Validate SL and TP Implementation
def test_stop_loss_take_profit():
    data = pd.DataFrame({
        'Close': [100, 105, 103],
        'High': [102, 107, 104],
        'Low': [99, 104, 100],
        'buy_signal': [1, 0, 0],
        'sell_signal': [0, 0, 1]
    })
    backtest = Backtest(data, 'long', 'atr', SL=1, SL_spike_out=True, TP=1, TS=False)
    backtest.run()
    assert backtest.info['exit_price_hist'][0] is not None, "Stop-loss or take-profit not executed"

# Test 3: Check Balance After Backtest
def test_balance_after_backtest():
    data = pd.DataFrame({
        'Close': [100, 102, 104],
        'High': [101, 103, 105],
        'Low': [99, 101, 103],
        'buy_signal': [1, 0, 0],
        'sell_signal': [0, 0, 1]
    })
    backtest = Backtest(data, 'long', 'atr', SL=2, SL_spike_out=True, TP=3, TS=False)
    backtest.run()
    assert backtest.balance >= 0, "Balance should not be negative"
