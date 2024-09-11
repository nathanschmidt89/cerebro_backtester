import pandas as pd
import numpy as np
import logging
from enum import Enum
from typing import Union, List, Optional


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enum for different stop-loss strategies
class SLType(Enum):
    ATR = 'atr'
    STANDARD_DEVIATION = 'sd'
    PERCENTAGE = 'pct'


# Exception for invalid parameters
class InvalidBacktestParameterError(Exception):
    pass


# Utility function to validate the DataFrame
def validate_backtest_data(data: pd.DataFrame):
    required_columns = ['Open', 'High', 'Low', 'Close', 'buy_signal', 'sell_signal']
    for column in required_columns:
        if column not in data.columns:
            raise InvalidBacktestParameterError(f"Missing required column: {column}")


# Moving average calculator (could be used for trend determination or filtering trades)
def moving_average(data: pd.Series, window: int):
    if window <= 0:
        raise ValueError("Window size must be greater than 0")
    return data.rolling(window=window).mean()


# Risk management configuration (could be extended with more complex logic)
class RiskConfig:
    def __init__(self, risk_pct: float, initial_balance: float, transaction_cost_unit: float, transaction_cost_top_pct: float):
        if not 0 < risk_pct <= 1:
            raise InvalidBacktestParameterError(f"Risk percentage must be between 0 and 1. Provided: {risk_pct}")
        self.risk_pct = risk_pct
        self.initial_balance = initial_balance
        self.transaction_cost_unit = transaction_cost_unit
        self.transaction_cost_top_pct = transaction_cost_top_pct

    def __str__(self):
        return f"RiskConfig(risk_pct={self.risk_pct}, initial_balance={self.initial_balance})"


# Transaction cost handler (to support various cost models)
class TransactionCostHandler:
    def __init__(self, unit_cost: float, max_pct_cost: float):
        self.unit_cost = unit_cost
        self.max_pct_cost = max_pct_cost

    def calculate_cost(self, trade_size: float, trade_value: float) -> float:
        """ Calculate the cost based on trade size and value. """
        return max(min(self.unit_cost * trade_size, trade_value * self.max_pct_cost), self.unit_cost)


# ATR or SD computation wrapped in a class for extensibility
class VolatilityCalculator:
    def __init__(self, data: pd.DataFrame, window: int = 14, method: str = 'atr'):
        self.data = data
        self.window = window
        self.method = method

    def compute(self) -> pd.Series:
        if self.method == 'atr':
            return self._calculate_atr()
        elif self.method == 'sd':
            return self._calculate_sd()
        else:
            raise InvalidBacktestParameterError(f"Unsupported volatility calculation method: {self.method}")

    def _calculate_atr(self) -> pd.Series:
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(self.window).mean()

    def _calculate_sd(self) -> pd.Series:
        return self.data['Close'].rolling(self.window).std()


# Complex strategy manager (managing multiple strategies or modifying parameters on the fly)
class StrategyManager:
    def __init__(self, strategies: List[dict]):
        self.strategies = strategies
        self.active_strategy = None

    def select_strategy(self, market_conditions: dict):
        """ Select the strategy based on market conditions (e.g., volatility, trend). """
        logger.info(f"Selecting strategy based on market conditions: {market_conditions}")
        # Placeholder logic; real implementation would consider multiple factors
        if market_conditions['volatility'] > 1.5:
            self.active_strategy = self.strategies[0]
        else:
            self.active_strategy = self.strategies[1]

    def apply_strategy(self):
        if not self.active_strategy:
            raise InvalidBacktestParameterError("No active strategy selected")
        logger.info(f"Applying strategy: {self.active_strategy}")
        # Apply the parameters of the selected strategy


# Higher-order functions (e.g., for customizing risk handling dynamically)
def dynamic_risk_adjustment(risk_pct: float):
    def adjust_risk(market_condition: dict):
        if market_condition['volatility'] > 1.5:
            return risk_pct * 1.2  # Increase risk
        else:
            return risk_pct * 0.8  # Decrease risk
    return adjust_risk


class Backtest:
    """ Backtest a trading strategy on historical data."""

    def __init__(self, info, side, SL_type, SL, SL_spike_out, TP, TS):
        self.info = info
        self.side = side
        self.SL_type = SL_type
        self.SL = SL
        self.SL_spike_out = SL_spike_out
        self.TP = TP
        self.TS = TS
        self.risk_pct = risk_pct
        self.fixed_size = fixed_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transaction_cost_unit = transaction_cost_unit
        self.transaction_cost_top_pct = transaction_cost_top_pct
        self.transaction_cost_unit_bottom = transaction_cost_unit_bottom

        if side == 'short':
            self.long_balance = 0
            self.short_balance = initial_balance
        elif side == 'long':
            self.long_balance = initial_balance
            self.short_balance = 0
        else:
            self.long_balance = initial_balance / 2
            self.short_balance = initial_balance / 2

        if SL_type == 'atr':
            self.info['atr'] = mtk.atr_calculator(self.info)
        elif SL_type == 'sd':
            self.info['sd'] = self.info['Close'].rolling(20).std()

    def run(self):
        long_shares = short_shares = trade_length = 0
        trade_idx = entry_date = entry_price = stop_loss = take_profit = stop_loss_org = atr = sd = None
        equity_curve, long_equity_curve, short_equity_curve, balance_curve, long_balance_curve, short_balance_curve, \
            long_shares_curve, short_shares_curve, trade_length_hist, trade_idx_hist, entry_price_hist, stop_price_hist, \
            take_profit_hist, stop_loss_org_hist, exit_reason_hist, exit_price_hist, trade_return_hist = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for col, default in [('buy_price', self.info['Close']), ('sell_price', self.info['Close']), ('buy_time', 'close'), ('sell_time', 'close')]:
            if col not in self.info.columns:
                self.info[col] = default

        if self.info['buy_time'].iloc[0] == 'open if trade':
            self.info.at[self.info.index[0], 'buy_signal'] = 0
        if self.info['sell_time'].iloc[0] == 'open if trade':
            self.info.at[self.info.index[0], 'sell_signal'] = 0

        for i in range(len(self.info)):
            buy_signal = self.info['buy_signal'].iloc[i]
            sell_signal = self.info['sell_signal'].iloc[i]
            buy_price = self.info['buy_price'].iloc[i]
            sell_price = self.info['sell_price'].iloc[i]
            # buy_time = self.info['buy_time'].iloc[i]
            sell_time = self.info['sell_time'].iloc[i]

            date = self.info.index[i]
            # open = self.info['Open'].iloc[i]
            high = self.info['High'].iloc[i]
            low = self.info['Low'].iloc[i]
            close = self.info['Close'].iloc[i]

            if self.SL_type == 'atr':
                atr = self.info['atr'].iloc[i] * self.SL if self.SL else self.info['atr'].iloc[i]
            elif self.SL_type == 'sd':
                sd = self.info['sd'].iloc[i] * self.SL if self.SL else self.info['sd'].iloc[i]

            # If there is an exit, exit details that need to be saved will not be restarted until now. In contrast, entry details are restarted upon exit.
            last_trade_idx = next((x for x in reversed(trade_idx_hist) if x), 0)
            exit_reason = exit_price = trade_return = None
            exited_long = exited_short = same_day_long_trade = same_day_short_trade = False

            same_day_trade_sell_first = False
            if buy_signal and sell_signal and sell_time == 'open':
                same_day_trade_sell_first = True

            while True:
                # LONGS
                if 'long' in self.side and not same_day_trade_sell_first:
                    new_long_shares = self.long_balance * self.risk_pct // buy_price
                    if new_long_shares > 0 and buy_signal and trade_length == 0:
                        new_long_shares = self.initial_balance * self.risk_pct // buy_price if fixed_size else new_long_shares
                        cost = new_long_shares * buy_price
                        fee = max(min(new_long_shares * self.transaction_cost_unit, cost * self.transaction_cost_top_pct),
                                  self.transaction_cost_unit_bottom)
                        total_cost = cost + fee
                        self.balance -= total_cost
                        self.long_balance -= total_cost
                        long_shares += new_long_shares
                        trade_length += 1
                        trade_idx = last_trade_idx + 1
                        entry_date = date
                        entry_price = buy_price
                        if self.SL:
                            if atr:
                                atr = atr if entry_price == close else self.info['atr'].iloc[i - 1] * self.SL
                                stop_loss = entry_price - atr
                            elif sd:
                                sd = sd if entry_price == close else self.info['sd'].iloc[i - 1] * self.SL
                                stop_loss = entry_price - sd
                            elif self.SL_type == 'pct' and self.SL:
                                stop_loss = entry_price * (1 - self.SL)
                        if self.TP:
                            take_profit = entry_price + atr * self.TP if atr else entry_price + sd * self.TP if sd else entry_price * (
                                    1 + self.SL * self.TP) if self.SL and self.SL_type == 'pct' else None
                            # There can be take_profit without stop_loss.
                        stop_loss_org = stop_loss

                    if long_shares > 0 and not exited_long and not exited_short:
                        if not date == entry_date:
                            trade_length += 1
                            if stop_loss:
                                comparable_stop_price = close if self.SL_spike_out else low
                                if comparable_stop_price <= stop_loss:
                                    exit_price = stop_loss
                                elif self.TS:
                                    new_stop_loss = high - atr if atr else high - sd if sd else high * (1 - self.SL) if self.SL and self.SL_type == 'pct' else None
                                    if new_stop_loss:
                                        stop_updated = True if new_stop_loss > stop_loss else False
                                        if stop_updated:
                                            stop_loss = new_stop_loss
                                            if close <= new_stop_loss:  # or low <= new_stop_loss and abs(high - close) > abs(low - close):
                                                exit_price = new_stop_loss
                            if not exit_price:  # To be conservative, stop will be hit before take_profit if same day.
                                if take_profit and high >= take_profit:
                                    exit_price = take_profit
                                elif sell_signal:
                                    exit_price = sell_price
                                    # Opposing signals are only executed if there is a trade to oppose.
                        elif sell_signal:
                            exit_price = sell_price
                            same_day_long_trade = True
                            # Only needed to break the loop not to Enter on another long trade again after just exited within same day.

                        if exit_price:
                            exited_long = True
                            exit_reason = 'SL' if exit_price == stop_loss_org else 'TS' if exit_price == stop_loss else 'TP' if exit_price == take_profit else 'Sell'
                            revenue = long_shares * exit_price
                            fee = max(min(long_shares * self.transaction_cost_unit, revenue * self.transaction_cost_top_pct), self.transaction_cost_unit_bottom)
                            net_revenue = revenue - fee
                            self.balance += net_revenue
                            self.long_balance += net_revenue
                            trade_return = exit_price / entry_price - 1 if entry_price else None
                            long_shares = 0
                            if not same_day_long_trade:
                                trade_length = 0
                                trade_idx = entry_date = entry_price = stop_loss = take_profit = stop_loss_org = None
                                # Final day trade_length=0, though exit might be at close. First day trade_length = 1 although entry might be at close.
                            else:
                                trade_length = 0
                # SHORTS
                if 'short' in self.side and not same_day_long_trade:
                    if same_day_trade_sell_first:
                        same_day_trade_sell_first = False
                    new_short_shares = self.short_balance * self.risk_pct // sell_price
                    if new_short_shares > 0 and sell_signal and trade_length == 0:
                        new_short_shares = self.initial_balance * self.risk_pct // sell_price if fixed_size else new_short_shares
                        revenue = new_short_shares * sell_price
                        fee = max(min(new_short_shares * self.transaction_cost_unit, revenue * self.transaction_cost_top_pct), self.transaction_cost_unit_bottom)
                        net_revenue = revenue - fee
                        self.balance += net_revenue
                        self.short_balance += net_revenue
                        short_shares += new_short_shares
                        trade_length += 1
                        trade_idx = last_trade_idx + 1
                        entry_date = date
                        entry_price = sell_price
                        if self.SL:
                            if atr:
                                atr = atr if entry_price == close else self.info['atr'].iloc[i - 1] * self.SL
                                stop_loss = entry_price + atr
                            elif sd:
                                sd = sd if entry_price == close else self.info['sd'].iloc[i - 1] * self.SL
                                stop_loss = entry_price + sd
                            elif self.SL_type == 'pct' and self.SL:
                                stop_loss = entry_price * (1 + self.SL)
                        if self.TP:
                            take_profit = entry_price - atr * self.TP if atr else entry_price - sd * self.TP if sd else entry_price * (
                                    1 - self.SL * self.TP) if self.SL and self.SL_type == 'pct' else None
                        stop_loss_org = stop_loss

                    if short_shares > 0:
                        if not date == entry_date:
                            trade_length += 1
                            if stop_loss:
                                comparable_stop_price = close if self.SL_spike_out else high
                                if comparable_stop_price >= stop_loss:
                                    exit_price = stop_loss
                                elif self.TS:
                                    new_stop_loss = low + atr if atr else low + sd if sd else low * (1 + self.SL) if self.SL and self.SL_type == 'pct' else None
                                    if new_stop_loss:
                                        stop_updated = True if new_stop_loss < stop_loss else False
                                        if stop_updated:
                                            stop_loss = new_stop_loss
                                            if close >= new_stop_loss:  # or high >= new_stop_loss and abs(high - close) < abs(low - close):
                                                exit_price = new_stop_loss
                            if not exit_price:
                                if take_profit and low <= take_profit:
                                    exit_price = take_profit
                                elif buy_signal:
                                    exit_price = buy_price
                        elif buy_signal:
                            exit_price = buy_price
                            same_day_short_trade = True

                        if exit_price and not exited_long and not exited_short:
                            # Here we would not need (not exited_short) since the situation in which we close a short
                            # and re-loop to open another short is not possible, we would only re-loop to open a long with long_shares > 0.
                            exited_short = True
                            exit_reason = 'SL' if exit_price == stop_loss_org else 'TS' if exit_price == stop_loss else 'TP' if exit_price == take_profit else 'Buy'
                            cost = short_shares * exit_price
                            fee = max(min(short_shares * self.transaction_cost_unit, cost * self.transaction_cost_top_pct), self.transaction_cost_unit_bottom)
                            total_cost = cost + fee
                            self.balance -= total_cost
                            self.short_balance -= total_cost
                            trade_return = -(exit_price / entry_price - 1) if entry_price else None
                            short_shares = 0
                            if not same_day_short_trade:
                                trade_length, trade_idx, entry_date, entry_price, stop_loss, take_profit, stop_loss_org = 0, None, None, None, None, None, None
                            else:
                                trade_length = 0

                if ((exited_long and not sell_time == 'open if trade') or
                        same_day_long_trade or same_day_short_trade or
                        not trade_length == 0 or not buy_signal or not self.long_balance * self.risk_pct // buy_price > 0):
                    break

            equity_curve.append(
                self.balance + long_shares * close - short_shares * close)
            long_equity_curve.append(self.long_balance + long_shares * close)
            short_equity_curve.append(self.short_balance - short_shares * close)
            balance_curve.append(self.balance)
            long_balance_curve.append(self.long_balance)
            short_balance_curve.append(self.short_balance)
            long_shares_curve.append(long_shares)
            short_shares_curve.append(short_shares)
            trade_length_hist.append(trade_length)
            trade_idx_hist.append(int(trade_idx) if trade_idx else None)
            entry_price_hist.append(entry_price)
            stop_price_hist.append(stop_loss)
            take_profit_hist.append(take_profit)
            stop_loss_org_hist.append(stop_loss_org)
            exit_reason_hist.append(exit_reason)
            exit_price_hist.append(exit_price)
            trade_return_hist.append(trade_return)

            if same_day_short_trade or same_day_long_trade:
                trade_length = 0
                trade_idx = entry_date = entry_price = stop_loss = take_profit = stop_loss_org = None

        self.info['equity_curve'] = equity_curve
        self.info['long_equity_curve'] = long_equity_curve
        self.info['short_equity_curve'] = short_equity_curve
        self.info['balance_curve'] = balance_curve
        self.info['long_balance_curve'] = long_balance_curve
        self.info['short_balance_curve'] = short_balance_curve
        self.info['long_shares_curve'] = long_shares_curve
        self.info['short_shares_curve'] = short_shares_curve
        running_max = np.maximum.accumulate(equity_curve)
        self.info['drawdown'] = (running_max - equity_curve) / running_max
        self.info['trade_length_hist'] = trade_length_hist
        self.info['trade_idx_hist'] = trade_idx_hist
        self.info['entry_price_hist'] = entry_price_hist
        self.info['stop_price_hist'] = stop_price_hist
        self.info['take_profit_hist'] = take_profit_hist
        self.info['stop_loss_org_hist'] = stop_loss_org_hist
        self.info['exit_reason_hist'] = exit_reason_hist
        self.info['exit_price_hist'] = exit_price_hist
        self.info['trade_return_hist'] = trade_return_hist
        with pd.option_context("future.no_silent_downcasting", True):
            # Had to include this to update pd behaviour due to a pd module that will be deprecated.
            self.info.replace([None], np.nan, inplace=True)
        return self.info


# Callback system for tracking real-time performance during backtest execution
class CallbackManager:
    def __init__(self):
        self.callbacks = []

    def register_callback(self, func):
        self.callbacks.append(func)

    def execute_callbacks(self, **kwargs):
        for callback in self.callbacks:
            callback(**kwargs)


def ev_as_return(info):
    winR = None
    RR = None
    EV = info['equity_curve'].iloc[-1] - info['equity_curve'].iloc[0]
    EVr = EV / info['equity_curve'].iloc[0]
    # = info_org['equity_curve'].iloc[-1] / info_org['equity_curve'].iloc[0] - 1
    return EVr, EV, winR, RR


# Advanced performance metrics calculator
class PerformanceMetrics:
    def __init__(self, equity_curve: pd.Series):
        self.equity_curve = equity_curve
        self.returns = equity_curve.pct_change().dropna()

    def sharpe_ratio(self, risk_free_rate: float = 0.01):
        excess_return = self.returns - risk_free_rate / 252
        return np.sqrt(252) * excess_return.mean() / excess_return.std()

    def max_drawdown(self):
        running_max = np.maximum.accumulate(self.equity_curve)
        drawdown = (running_max - self.equity_curve) / running_max
        return drawdown.max()

    def sortino_ratio(self, risk_free_rate: float = 0.01):
        downside_return = self.returns[self.returns < 0] - risk_free_rate / 252
        return np.sqrt(252) * self.returns.mean() / downside_return.std()


def ev_calculator(info, SL_type):
    """
    Calculate the Expected Value (EV) of a trading strategy.
    EV is the Expected Value of a trade, which is the Weighted Average dollar Return per trade.
    EVr is equivalent to the Weighted Average Reward per trade (WART), and it's equal to EV / R(not win).
    Outer scope variables ok.
    """

    if info['trade_length_hist'].mean() == 0 and not info[~info['exit_price_hist'].isna()].empty:
        # Looking for same_day_trades.
        info['win'] = np.where(info['buy_time'] == 'open',
                               np.where(info['exit_price_hist'] > info['entry_price_hist'], 1, 0),
                               np.where(info['exit_price_hist'] < info['entry_price_hist'], 1, 0))
        info['R'] = np.where(info['buy_time'] == 'open',
                             info['exit_price_hist'] - info['entry_price_hist'],
                             info['entry_price_hist'] - info['exit_price_hist'])
    else:
        info['exit_price_hist_f1'] = info['exit_price_hist'].shift(-1)
        mask = info['exit_price_hist_f1'].isna()
        info.loc[~mask, 'win'] = np.where(info.loc[~mask, 'long_shares_curve'] > 0,
                                          np.where(info.loc[~mask, 'exit_price_hist_f1'] > info.loc[
                                              ~mask, 'entry_price_hist'], 1, 0),
                                          np.where(info.loc[~mask, 'exit_price_hist_f1'] < info.loc[
                                              ~mask, 'entry_price_hist'], 1, 0))
        info.loc[~mask, 'R'] = np.where(info.loc[~mask, 'long_shares_curve'] > 0,
                                        info.loc[~mask, 'exit_price_hist_f1'] - info.loc[~mask, 'entry_price_hist'],
                                        info.loc[~mask, 'entry_price_hist'] - info.loc[~mask, 'exit_price_hist_f1'])
        info['win'] = info['win'].shift(1)
        info['R'] = info['R'].shift(1)
        info.drop(columns=['exit_price_hist_f1'], inplace=True)

    info.loc[info['exit_price_hist'].isna(), 'win'] = np.nan

    if info['trade_length_hist'].iloc[-1] != 0:
        if info['long_shares_curve'].iloc[-1] > 0:
            info.at[info.index[-1], 'win'] = int(info['entry_price_hist'].iloc[-1] < info['Close'].iloc[-1])
            info.at[info.index[-1], 'R'] = info['Close'].iloc[-1] - info['entry_price_hist'].iloc[-1]
        elif info['short_shares_curve'].iloc[-1] > 0:
            info.at[info.index[-1], 'win'] = int(info['entry_price_hist'].iloc[-1] > info['Close'].iloc[-1])
            info.at[info.index[-1], 'R'] = info['entry_price_hist'].iloc[-1] - info['Close'].iloc[-1]

    if info['win'].value_counts().empty and info['trade_length_hist'].mean() == 0:
        EVr = EV = winR = RR = None
        return EVr, winR, RR
    elif info['win'].value_counts().empty:
        EVr, EV, winR, RR = ev_as_return(info)
    elif len(info['win'].value_counts()) == 1:
        winR = info['win'].value_counts().index[0]
        if SL_type:
            SL_distance = (info[info['trade_length_hist'] == 1]['entry_price_hist'] - info[info['trade_length_hist'] == 1]['stop_price_hist']).mean()
            RR = info[info['R'] > 0]['R'].mean() / SL_distance if winR == 1 else None
            EVr = RR if winR == 1 else info[info['R'] < 0]['R'].mean() / SL_distance
        else:
            RR = info[info['R'] > 0]['R'].mean() if winR == 1 else None
            EVr = info[info['R'] > 0]['R'].mean() if winR == 1 else info[info['R'] < 0]['R'].mean()
    else:
        winR = info['win'].value_counts()[1] / info['win'].count()
        EV = winR * info[info['R'] > 0]['R'].mean() + (1 - winR) * info[info['R'] < 0]['R'].mean()
        EVr = EV / abs(info[info['R'] < 0]['R'].mean())
        RR = abs(info[info['R'] > 0]['R'].mean() / info[info['R'] < 0]['R'].mean())  # EVr = winR * RR - (1 - winR)

    EVr = round(EVr, 4)
    winR = round(winR, 2) if winR else winR
    RR = round(RR, 2) if RR else RR

    return EVr, winR, RR


def trade_details_generator(info):
    info['open2close'] = info['Close'] / info['Open'] - 1
    info['open2high'] = info['High'] / info['Open'] - 1
    info['open2low'] = info['Low'] / info['Open'] - 1

    info_either = info[(info['win'] == 1) | (info['win'] == 0)]
    open2close = info_either['open2close'].mean()
    open2high = info_either['open2high'].mean()
    open2low = info_either['open2low'].mean()
    open2close_max = info_either['open2close'].max()
    open2high_max = info_either['open2high'].max()
    open2low_max = info_either['open2low'].max()
    open2close_min = info_either['open2close'].min()
    open2high_min = info_either['open2high'].min()
    open2low_min = info_either['open2low'].min()

    info_win = info[info['win'] == 1]
    open2close_win = info_win['open2close'].mean()
    open2high_win = info_win['open2high'].mean()
    open2low_win = info_win['open2low'].mean()
    open2close_win_max = info_win['open2close'].max()
    open2high_win_max = info_win['open2high'].max()
    open2low_win_max = info_win['open2low'].max()
    open2close_win_min = info_win['open2close'].min()
    open2high_win_min = info_win['open2high'].min()
    open2low_win_min = info_win['open2low'].min()

    info_noWin = info[info['win'] == 0]
    open2close_noWin = info_noWin['open2close'].mean()
    open2high_noWin = info_noWin['open2high'].mean()
    open2low_noWin = info_noWin['open2low'].mean()
    open2close_noWin_max = info_noWin['open2close'].max()
    open2high_noWin_max = info_noWin['open2high'].max()
    open2low_noWin_max = info_noWin['open2low'].max()
    open2close_noWin_min = info_noWin['open2close'].min()
    open2high_noWin_min = info_noWin['open2high'].min()
    open2low_noWin_min = info_noWin['open2low'].min()

    return_list = [
        open2close, open2high, open2low,
        open2close_max, open2high_max, open2low_max,
        open2close_min, open2high_min, open2low_min,
        open2close_win, open2high_win, open2low_win,
        open2close_win_max, open2high_win_max, open2low_win_max,
        open2close_win_min, open2high_win_min, open2low_win_min,
        open2close_noWin, open2high_noWin, open2low_noWin,
        open2close_noWin_max, open2high_noWin_max, open2low_noWin_max,
        open2close_noWin_min, open2high_noWin_min, open2low_noWin_min
    ]
    return return_list


def compute_ratios(measure, info, idx, idx2=None, None_transform=True):
    """ Calculate the average of the ratios based on their position on the return of the backtest_run()."""
    if idx2:
        ratios = [item[idx2][idx] for item in info]
    else:
        ratios = [item[idx] for item in info]

    if None_transform:
        # Assuming if: None are transformed to zero values.
        ratios = [0 if isinstance(_, float) and np.isnan(_) or _ is None else _ for _ in ratios]
    else:
        ratios = [None if isinstance(_, float) and np.isnan(_) else _ for _ in ratios]
        if all(i is None for i in ratios):
            return None

    if measure == 'max':
        result = np.max([_ for _ in ratios if _ is not None])
    elif measure == 'min':
        result = np.min([_ for _ in ratios if _ is not None])
    else:
        result = np.mean([_ for _ in ratios if _ is not None])

    return result


class PortfolioRebalancer:
    def __init__(self, initial_weights: dict, risk_model: Union[RiskConfig, callable]):
        self.weights = initial_weights
        self.risk_model = risk_model

    def rebalance(self, market_conditions: dict):
        """ Adjust portfolio weights based on risk model and market conditions """
        logger.info(f"Rebalancing portfolio with market conditions: {market_conditions}")
        if isinstance(self.risk_model, RiskConfig):
            # Static risk model
            adjusted_risk_pct = self.risk_model.risk_pct
        elif callable(self.risk_model):
            # Dynamic risk adjustment
            adjusted_risk_pct = self.risk_model(market_conditions)
        else:
            raise InvalidBacktestParameterError("Unsupported risk model type")

        for asset in self.weights:
            self.weights[asset] *= adjusted_risk_pct
        logger.info(f"New portfolio weights: {self.weights}")


def global_content(content, strat_name, attributes):
    if not periods_for_split:
        return content
    else:
        period_start = attributes[0][order_return_backtest_run['period_start']]
        period_end = attributes[-1][order_return_backtest_run['period_end']]
        side = attributes[0][order_return_backtest_run['side']]
        SL_type = attributes[0][order_return_backtest_run['SL_type']]
        SL = attributes[0][order_return_backtest_run['SL']]
        TP = attributes[0][order_return_backtest_run['TP']]
        avg_total_return = compute_ratios('mean', attributes, order_return_backtest_run['total_return'])
        max_total_return = compute_ratios('max', attributes, order_return_backtest_run['total_return'])
        min_total_return = compute_ratios('min', attributes, order_return_backtest_run['total_return'])
        avg_adj_total_return = compute_ratios('mean', attributes, order_return_backtest_run['adj_total_return'])
        max_adj_total_return = compute_ratios('max', attributes, order_return_backtest_run['adj_total_return'])
        min_adj_total_return = compute_ratios('min', attributes, order_return_backtest_run['adj_total_return'])
        avg_max_drawdown = compute_ratios('mean', attributes, order_return_backtest_run['max_drawdown'])
        max_max_drawdown = compute_ratios('max', attributes, order_return_backtest_run['max_drawdown'])
        min_max_drawdown = compute_ratios('min', attributes, order_return_backtest_run['max_drawdown'])
        avg_no_trades = compute_ratios('mean', attributes, order_return_backtest_run['no_trades'])
        max_no_trades = compute_ratios('max', attributes, order_return_backtest_run['no_trades'])
        min_no_trades = compute_ratios('min', attributes, order_return_backtest_run['no_trades'])
        avg_trading_frequency = compute_ratios('mean', attributes, order_return_backtest_run['trading_frequency'])
        max_trading_frequency = compute_ratios('max', attributes, order_return_backtest_run['trading_frequency'])
        min_trading_frequency = compute_ratios('min', attributes, order_return_backtest_run['trading_frequency'])
        avg_occupancy = compute_ratios('mean', attributes, order_return_backtest_run['occupancy'])
        max_occupancy = compute_ratios('max', attributes, order_return_backtest_run['occupancy'])
        min_occupancy = compute_ratios('min', attributes, order_return_backtest_run['occupancy'])
        avg_sharpe_ratio = compute_ratios('mean', attributes, order_return_backtest_run['sharpe_ratio'])
        max_sharpe_ratio = compute_ratios('max', attributes, order_return_backtest_run['sharpe_ratio'])
        min_sharpe_ratio = compute_ratios('min', attributes, order_return_backtest_run['sharpe_ratio'])
        avg_long_sharpe_ratio = compute_ratios('mean', attributes, order_return_backtest_run['long_sharpe_ratio'])
        max_long_sharpe_ratio = compute_ratios('max', attributes, order_return_backtest_run['long_sharpe_ratio'])
        min_long_sharpe_ratio = compute_ratios('min', attributes, order_return_backtest_run['long_sharpe_ratio'])
        avg_short_sharpe_ratio = compute_ratios('mean', attributes, order_return_backtest_run['short_sharpe_ratio'])
        max_short_sharpe_ratio = compute_ratios('max', attributes, order_return_backtest_run['short_sharpe_ratio'])
        min_short_sharpe_ratio = compute_ratios('min', attributes, order_return_backtest_run['short_sharpe_ratio'])
        avg_adj_sharpe_ratio = compute_ratios('mean', attributes, order_return_backtest_run['adj_sharpe_ratio'], None_transform=False)
        max_adj_sharpe_ratio = compute_ratios('max', attributes, order_return_backtest_run['adj_sharpe_ratio'])
        min_adj_sharpe_ratio = compute_ratios('min', attributes, order_return_backtest_run['adj_sharpe_ratio'])
        avg_ev_ratio = compute_ratios('mean', attributes, order_return_backtest_run['ev_ratio'], None_transform=False)
        max_ev_ratio = compute_ratios('max', attributes, order_return_backtest_run['ev_ratio'])
        min_ev_ratio = compute_ratios('min', attributes, order_return_backtest_run['ev_ratio'])
        avg_winR = compute_ratios('mean', attributes, order_return_backtest_run['winR'], None_transform=False)
        max_winR = compute_ratios('max', attributes, order_return_backtest_run['winR'])
        min_winR = compute_ratios('min', attributes, order_return_backtest_run['winR'])
        avg_RR = compute_ratios('mean', attributes, order_return_backtest_run['RR'], None_transform=False)
        max_RR = compute_ratios('max', attributes, order_return_backtest_run['RR'])
        min_RR = compute_ratios('min', attributes, order_return_backtest_run['RR'])

        content += (
            f'\nSTRATEGY {strat_name}\n'
            f"Averages across {len(attributes)} {str(split_years) + ' ' if periods_for_split == 'yearly' else ''}{periods_for_split} period splits -------->\n"
            f"Period: {mtk.date_to_string(period_start)} to {mtk.date_to_string(period_end)}\n"
        )
        var = {
            "Direction": side,
            "Stop type": SL_type,
            "Stop Amt": SL,
            "Take Profit Prop": TP,
            "Total Return Pct": (
                str(mtk.string_percent(avg_total_return)) if not avg_total_return else
                str(mtk.string_percent(avg_total_return)) + ' ~ ' +
                str(mtk.string_percent(min_total_return)) + ' / ' +
                str(mtk.string_percent(max_total_return))
            ),
            "Adj Total Return Pct": (
                str(mtk.string_percent(avg_adj_total_return)) if not avg_adj_total_return else
                str(mtk.string_percent(avg_adj_total_return)) + ' ~ ' +
                str(mtk.string_percent(min_adj_total_return)) + ' / ' +
                str(mtk.string_percent(max_adj_total_return))
            ),
            "Max Drawdown Pct": (
                str(mtk.string_percent(avg_max_drawdown)) if not avg_max_drawdown else
                str(mtk.string_percent(avg_max_drawdown)) + ' ~ ' +
                str(mtk.string_percent(min_max_drawdown)) + ' / ' +
                str(mtk.string_percent(max_max_drawdown))
            ),
            "No. Trades": (
                str(avg_no_trades) if not avg_no_trades else
                str(int(round(avg_no_trades, 0))) + ' ~ ' +
                str(int(round(min_no_trades, 0))) + ' / ' +
                str(int(round(max_no_trades, 0)))
            ),
            "Trading Frequency": (
                str(avg_trading_frequency) if not avg_trading_frequency else
                str(int(round(avg_trading_frequency, 0))) + ' ~ ' +
                str(int(round(min_trading_frequency, 0))) + ' / ' +
                str(int(round(max_trading_frequency, 0)))
            ),
            "Occupancy Pct": (
                str(mtk.string_percent(avg_occupancy)) if not avg_occupancy else
                str(mtk.string_percent(avg_occupancy)) + ' ~ ' +
                str(mtk.string_percent(min_occupancy)) + ' / ' +
                str(mtk.string_percent(max_occupancy))
            ),
            "Sharpe ratio": (
                str(avg_sharpe_ratio) if not avg_sharpe_ratio else
                str(round(avg_sharpe_ratio, 2)) + ' ~ ' +
                str(round(min_sharpe_ratio, 2)) + ' / ' +
                str(round(max_sharpe_ratio, 2))
            ),
            "Long Sharpe ratio": (
                str(avg_long_sharpe_ratio) if not avg_long_sharpe_ratio else
                str(round(avg_long_sharpe_ratio, 2)) + ' ~ ' +
                str(round(min_long_sharpe_ratio, 2)) + ' / ' +
                str(round(max_long_sharpe_ratio, 2))
            ),
            "Short Sharpe ratio": (
                str(avg_short_sharpe_ratio) if not avg_short_sharpe_ratio else
                str(round(avg_short_sharpe_ratio, 2)) + ' ~ ' +
                str(round(min_short_sharpe_ratio, 2)) + ' / ' +
                str(round(max_short_sharpe_ratio, 2))
            ),
            "Adj Sharpe ratio": (
                str(avg_adj_sharpe_ratio) if not avg_adj_sharpe_ratio else
                str(round(avg_adj_sharpe_ratio, 2)) + ' ~ ' +
                str(round(min_adj_sharpe_ratio, 2)) + ' / ' +
                str(round(max_adj_sharpe_ratio, 2))
            ),
            "EV ratio": (
                str(avg_ev_ratio) if not avg_ev_ratio else
                str(round(avg_ev_ratio, 2)) + ' ~ ' +
                str(round(min_ev_ratio, 2)) + ' / ' +
                str(round(max_ev_ratio, 2))
            ),
            "Win ratio Pct": (
                str(mtk.string_percent(avg_winR)) if not avg_winR else
                str(mtk.string_percent(avg_winR)) + ' ~ ' +
                str(mtk.string_percent(min_winR)) + ' / ' +
                str(mtk.string_percent(max_winR))
            ),
            "RR": (
                str(avg_RR) if not avg_RR else
                str(round(avg_RR, 2)) + ' ~ ' +
                str(round(min_RR, 2)) + ' / ' +
                str(round(max_RR, 2))
            ),
        }
        variables_if_daytrading = {}
        if strat_name.startswith('Day'):
            avg_open2close_win = compute_ratios('mean', attributes,
                                                order_return_trade_details_generator['open2close_win'],
                                                order_return_backtest_run['trade_details'],
                                                None_transform=False)
            avg_open2high_win = compute_ratios('mean', attributes,
                                               order_return_trade_details_generator['open2high_win'],
                                               order_return_backtest_run['trade_details'],
                                               None_transform=False)
            avg_open2low_win = compute_ratios('mean', attributes,
                                              order_return_trade_details_generator['open2low_win'],
                                              order_return_backtest_run['trade_details'],
                                              None_transform=False)
            avg_open2close_win_max = compute_ratios('max', attributes,
                                                    order_return_trade_details_generator['open2close_win'],
                                                    order_return_backtest_run['trade_details'])
            avg_open2high_win_max = compute_ratios('max', attributes,
                                                   order_return_trade_details_generator['open2high_win'],
                                                   order_return_backtest_run['trade_details'])
            avg_open2low_win_max = compute_ratios('max', attributes,
                                                  order_return_trade_details_generator['open2low_win'],
                                                  order_return_backtest_run['trade_details'])
            avg_open2close_win_min = compute_ratios('min', attributes,
                                                    order_return_trade_details_generator['open2close_win'],
                                                    order_return_backtest_run['trade_details'])
            avg_open2high_win_min = compute_ratios('min', attributes,
                                                   order_return_trade_details_generator['open2high_win'],
                                                   order_return_backtest_run['trade_details'])
            avg_open2low_win_min = compute_ratios('min', attributes,
                                                  order_return_trade_details_generator['open2low_win'],
                                                  order_return_backtest_run['trade_details'])
            variables_if_daytrading = {
                "Avg. Open2Close Win": str(mtk.string_percent(avg_open2close_win)) + ' ~ ' + str(
                    mtk.string_percent(avg_open2close_win_min)) + ' / ' + str(
                    mtk.string_percent(avg_open2close_win_max)),
                "Avg. Open2High Win": str(mtk.string_percent(avg_open2high_win)) + ' ~ ' + str(
                    mtk.string_percent(avg_open2high_win_min)) + ' / ' + str(
                    mtk.string_percent(avg_open2high_win_max)),
                "Avg. Open2Low Win": str(mtk.string_percent(avg_open2low_win)) + ' ~ ' + str(
                    mtk.string_percent(avg_open2low_win_min)) + ' / ' + str(
                    mtk.string_percent(avg_open2low_win_max)),
            }
        content = variables_to_content(content, {**var, **variables_if_daytrading})
    return content


# Integration of Monte Carlo simulations for stress testing
class MonteCarloSimulator:
    def __init__(self, backtest_instance: Backtest, num_simulations: int = 1000):
        self.backtest = backtest_instance
        self.num_simulations = num_simulations

    def simulate(self):
        logger.info(f"Running {self.num_simulations} Monte Carlo simulations")
        simulated_equity_curves = []
        for _ in range(self.num_simulations):
            random_noise = np.random.normal(0, 0.01, len(self.backtest.info))
            simulated_returns = self.backtest.info['equity_curve'].pct_change().fillna(0) + random_noise
            simulated_equity_curve = (1 + simulated_returns).cumprod() * self.backtest.initial_balance
            simulated_equity_curves.append(simulated_equity_curve)

        return pd.DataFrame(simulated_equity_curves).T


def variables_to_content(content, var):
    for title, value in var.items():
        if value:
            if isinstance(value, float):
                content += f"{title}: {value:.2f}\n"
            else:
                content += f"{title}: {value}\n"
        else:
            content += f"{title}: {value}\n"
    return content


def individual_variables(strat_name, attributes):
    side = attributes[order_return_backtest_run['side']]
    SL_type = attributes[order_return_backtest_run['SL_type']]
    SL = attributes[order_return_backtest_run['SL']]
    TP = attributes[order_return_backtest_run['TP']]
    total_return = attributes[order_return_backtest_run['total_return']]
    adj_total_return = attributes[order_return_backtest_run['adj_total_return']]
    max_drawdown = attributes[order_return_backtest_run['max_drawdown']]
    no_trades = attributes[order_return_backtest_run['no_trades']]
    trading_frequency = attributes[order_return_backtest_run['trading_frequency']]
    occupancy = attributes[order_return_backtest_run['occupancy']]
    sharpe_ratio = attributes[order_return_backtest_run['sharpe_ratio']]
    long_sharpe_ratio = attributes[order_return_backtest_run['long_sharpe_ratio']]
    short_sharpe_ratio = attributes[order_return_backtest_run['short_sharpe_ratio']]
    adj_sharpe_ratio = attributes[order_return_backtest_run['adj_sharpe_ratio']]
    ev_ratio = attributes[order_return_backtest_run['ev_ratio']]
    winR = attributes[order_return_backtest_run['winR']]
    RR = attributes[order_return_backtest_run['RR']]
    trade_details = attributes[order_return_backtest_run['trade_details']]

    variables_if_period_split = {}
    if not periods_for_split:
        variables_if_period_split = {
            "Direction": side,
            "Stop type": SL_type,
            "Stop Amt": SL,
            "Take Profit Prop": TP,
        }
    variables = {
        "Total Return": mtk.string_percent(total_return),
        "Adj Total Return": mtk.string_percent(adj_total_return),
        "Max Drawdown": mtk.string_percent(max_drawdown),
        'No. Trades': int(round(no_trades, 0)) if no_trades else None,
        'Trading Frequency': int(round(trading_frequency, 0)) if trading_frequency else None,
        'Occupancy': mtk.string_percent(occupancy),
        "Sharpe ratio": sharpe_ratio,
        "Long Sharpe ratio": long_sharpe_ratio,
        "Short Sharpe ratio": short_sharpe_ratio,
        "Adj Sharpe ratio": adj_sharpe_ratio,
        "EV ratio": ev_ratio,
        "Win ratio": mtk.string_percent(winR),
        "RR": RR,
    }
    variables_if_daytrading = {}
    if strat_name.startswith('Day'):
        open2close_win = trade_details[order_return_trade_details_generator['open2close_win']]
        open2high_win = trade_details[order_return_trade_details_generator['open2high_win']]
        open2low_win = trade_details[order_return_trade_details_generator['open2low_win']]
        open2close_win_max = trade_details[order_return_trade_details_generator['open2close_win_max']]
        open2high_win_max = trade_details[order_return_trade_details_generator['open2high_win_max']]
        open2low_win_max = trade_details[order_return_trade_details_generator['open2low_win_max']]
        open2close_win_min = trade_details[order_return_trade_details_generator['open2close_win_min']]
        open2high_win_min = trade_details[order_return_trade_details_generator['open2high_win_min']]
        open2low_win_min = trade_details[order_return_trade_details_generator['open2low_win_min']]
        variables_if_daytrading = {
            "Open2Close Win": str(mtk.string_percent(open2close_win)) + ' ~ ' + str(
                mtk.string_percent(open2close_win_min)) + ' / ' + str(
                mtk.string_percent(open2close_win_max)),
            "Open2High Win": str(mtk.string_percent(open2high_win)) + ' ~ ' + str(
                mtk.string_percent(open2high_win_min)) + ' / ' + str(
                mtk.string_percent(open2high_win_max)),
            "Open2Low Win": str(mtk.string_percent(open2low_win)) + ' ~ ' + str(
                mtk.string_percent(open2low_win_min)) + ' / ' + str(
                mtk.string_percent(open2low_win_max)),
        }
    return {**variables_if_period_split, **variables, **variables_if_daytrading}


# Sample test cases
def test_backtest_long():
    data = pd.DataFrame({
        'Close': np.random.normal(100, 5, 100),
        'High': np.random.normal(102, 5, 100),
        'Low': np.random.normal(98, 5, 100),
        'buy_signal': np.random.randint(0, 2, 100),
        'sell_signal': np.random.randint(0, 2, 100),
    })
    bt = Backtest(
        info=data,
        side='long',
        SL_type='atr',
        SL=1.5,
        SL_spike_out=True,
        TP=2,
        TS=0.5
    )
    bt.run()
    print("Long side backtest complete!")


def test_backtest_short():
    data = pd.DataFrame({
        'Close': np.random.normal(100, 5, 100),
        'High': np.random.normal(102, 5, 100),
        'Low': np.random.normal(98, 5, 100),
        'buy_signal': np.random.randint(0, 2, 100),
        'sell_signal': np.random.randint(0, 2, 100),
    })
    bt = Backtest(
        info=data,
        side='short',
        SL_type='atr',
        SL=1.5,
        SL_spike_out=False,
        TP=1.5,
        TS=1
    )
    bt.run()
    print("Short side backtest complete!")


# Example of callback to log equity curve progress
def log_equity_curve_update(equity_curve):
    logger.info(f"Current equity curve value: {equity_curve[-1]}")


# Register and execute callbacks during backtest
callback_manager = CallbackManager()
callback_manager.register_callback(log_equity_curve_update)


# Run all tests
if __name__ == '__main__':
    test_backtest_long()
    test_backtest_short()
