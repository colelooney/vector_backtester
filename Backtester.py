import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Abstract base class for a trading strategy.
    All strategy classes must inherit from this class and implement the generate_signals method.
    """
    @abstractmethod
    def generate_signals(self,data):
        """
        generates trading signals for the given dataset
        """
        pass

class MovingAverageStrategy(Strategy):
    def __init__(self, short_window = 50, long_window = 200):
        self.short_window = short_window
        self.long_window = long_window  

    def generate_signals(self, data):
        """Generates signals based on the moving average crossover logic."""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close'].copy()  # Use 'Close' prices for signals
        
        signals['short_ma'] = signals['price'].rolling(window=self.short_window).mean()
        signals['long_ma'] = signals['price'].rolling(window=self.long_window).mean()
        
        signals['signal'] = np.where(signals['short_ma'] > signals['long_ma'], 1, 0)
        
        return signals['signal']

class MomentumStrategy(Strategy):
    def __init__(self, momentum_window=90):
        self.momentum_window = momentum_window

    def generate_signals(self, data):
        """Generates signals based on price momentum."""
        signals = pd.DataFrame(index=data.index)
        
        signals['momentum'] = data['Close'].pct_change(periods=self.momentum_window)
        
        signals['signal'] = np.where(signals['momentum'] > 0, 1, 0)
        
        return signals['signal']


class VectorizedBacktester:
    """
    A vectorized backtesting framework for trading strategies.

    Attributes:
        symbol (str): The stock symbol to backtest.
        start_date (str): The start date for historical data in 'YYYY-MM-DD' format.
        end_date (str): The end date for historical data in 'YYYY-MM-DD' format.
        data (pd.DataFrame): DataFrame holding the historical price data.
        results (pd.DataFrame): DataFrame holding the backtest results.
    """
    
    def __init__(self,symbol,start_date,end_date,commission=0.001):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        # self.data, = self._load_data()
        self.data = self._load_data_csv()
        self.results = None
        self.commission = commission 
    
    # def _load_data(self):
    #     """
    #     Loads historical data using yahoo finance library
    #     """
    #     df = yf.download(self.symbol,start = self.start_date, end = self.end_date)
    #     data  = df[['Close']].copy() #Take close column
    #     data.rename(columns = {'Close': 'price'}, inplace = True)
    #     return data, df

    def _load_data_csv(self):
        """
        Loads historical data from a CSV file.
        """
        data = pd.read_csv(r"C:\Users\colel\OneDrive\Documents\[03] Professional\Projects\HSI_Prices.csv")
        return data
    
    def run(self,strategy_object):
        """
        runs a backtest given a strategy object.
        """
        print(f"Running backtest with strategy: {strategy_object.__class__.__name__}...")

        signals = strategy_object.generate_signals(self.data)

        results = self.data.copy()
        results['signals'] = signals

        results['position'] = results['signals'].shift(1)

        results['position'].fillna(0, inplace=True)
        results['position'] = results['position'].astype(int)

        results['market_returns'] = results['Close'].pct_change()
        results['strategy_returns'] = results['market_returns'] * results['position'] - results['position'].diff().abs() * self.commission

        results['cumulative_market'] = (1 + results['market_returns']).cumprod()
        results['cumulative_strategy'] = (1 + results['strategy_returns']).cumprod()

        self.results = results.dropna()
        print("Backtest complete.")

    
    def plot_results(self, strategy_object):
        """Plots the cumulative returns of the strategy vs. buy-and-hold."""
        if self.results is None:
            print("Please run the strategy first.")
            return
            
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 7))
        self.results[['cumulative_market', 'cumulative_strategy']].plot(ax=plt.gca())
        plt.title(f'{strategy_object.__class__.__name__}: {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend(['Buy and Hold', 'Strategy'])
        plt.show()

    def calculate_metrics(self):
        """
        Calculates and prints key performance metrics.
        """
        if self.results is None:
            print("Please run the strategy first.")
            return

        def get_max_drawdown(cum_returns):
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            return drawdown.min()

        metrics = {}
        
        metrics['Total Strategy Return (%)'] = (self.results['cumulative_strategy'].iloc[-1] - 1) * 100
        metrics['Total Market Return (%)'] = (self.results['cumulative_market'].iloc[-1] - 1) * 100


        metrics['Annualized Strategy Volatility (%)'] = self.results['strategy_returns'].std() * np.sqrt(252) * 100
        
        #Adjust for risk_free_rate
        annualized_return = self.results['strategy_returns'].mean() * 252
        annualized_vol = self.results['strategy_returns'].std() * np.sqrt(252)
        metrics['Annualized Sharpe Ratio'] = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        metrics['Max Strategy Drawdown (%)'] = get_max_drawdown(self.results['cumulative_strategy']) * 100
        metrics['Max Market Drawdown (%)'] = get_max_drawdown(self.results['cumulative_market']) * 100
        
        print("\n--- Performance Metrics ---")
        for key, value in metrics.items():
            print(f"{key:<35}: {value:.2f}")
        print("---------------------------\n")



if __name__ == '__main__':
    symbol = '^HSI'  # Hang Seng Index
    start = '2020-01-01'
    end = '2025-06-09'

    backtester = VectorizedBacktester(symbol = symbol, start_date = start, end_date = end)

    ma = MovingAverageStrategy(short_window= 50, long_window= 200)
    
    backtester.run(ma)
    backtester.plot_results(ma)
    backtester.calculate_metrics()

    momentum_strategy = MomentumStrategy(momentum_window=100)

    backtester.run(momentum_strategy)
    backtester.plot_results(momentum_strategy)
    backtester.calculate_metrics()

