# __init__.py

"""
A backtesting framework for trading strategies.
"""

from .backtest import Backtest, BacktestError
from .utils import RiskConfig, TransactionCostHandler, VolatilityCalculator, StrategyManager
from .metrics import PerformanceMetrics
from .risk_management import PortfolioRebalancer
from .simulations import MonteCarloSimulator
from .callbacks import CallbackManager

__all__ = [
    'Backtest',
    'BacktestError',
    'RiskConfig',
    'TransactionCostHandler',
    'VolatilityCalculator',
    'StrategyManager',
    'PerformanceMetrics',
    'PortfolioRebalancer',
    'MonteCarloSimulator',
    'CallbackManager'
]

# Optional: Display package information on import
import logging

logger = logging.getLogger(__name__)
logger.info("Package 'backtest_framework' loaded. Ready to use!")


# Optional: Initial setup or configuration code
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    logger.info(f"Logging setup complete with level: {logging.getLevelName(level)}")


setup_logging()


# Optional: Example configuration or package initialization
def initialize_package(config=None):
    if config:
        logger.info(f"Initializing package with config: {config}")
    else:
        logger.info("Initializing package with default settings")

    # Add any additional initialization logic here

