import datetime as dt
import numpy as np
import pandas as pd
from util import get_data, plot_data


def author():
    return "axiao31"

def compute_portvals(df_trades, sd, ed, start_val=100000,  commission=9.95,impact=0.005):    
    
    orders = df_trades
    orders.sort_index(inplace=True)
    sym = list(df_trades.columns)

    dr = pd.date_range(sd, ed)

   #Getting prices for stocks
    prices = get_data(sym, dr)
    prices.ffill(inplace=True)   
    prices.bfill(inplace=True)    

    #Delete SPY if not in Symbols
    if 'SPY' not in sym:
        prices.drop('SPY', axis=1, inplace=True)

    #prices['CASH'] = pd.Series(np.ones(shape=prices.shape[0]), index=prices.index)
    # trade = prices.copy()*0.0
    prices['Cash'] = 1   #Adding a cash field to prices and initializing to 1 so that we can multiply it straight away with holdings
    trade = pd.DataFrame(np.zeros(prices.shape), columns=prices.columns, index=prices.index)
    trade.iloc[0, -1] = start_val  #Cash for first date set to start value
 
    for s in sym:
        for index, row in orders.iterrows():
            price = prices.loc[index, s]  
            t_cost = commission + price * row[s]

            if row[s]< 0:
                trade.loc[index, s] = trade.loc[index, s] + row[s]
                price = price - (price * impact)
            else:
                trade.loc[index, s] = trade.loc[index, s] + row[s]
                price = price + (price * impact)

            # Calculating the cash after cost
            trade.loc[index, 'Cash'] -= t_cost

        # compute total value
        holding = trade.cumsum()
        c_sum = holding * prices  
        portvals = c_sum.sum(axis=1)

    return portvals

# Calculate cumulative return, standard daily return, average daily return and sharpe ratio
def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    avg_daily_ret=0
    std_daily_ret=0
    sharpe_ratio=0

    daily_returns = (port_val / port_val.shift(1)) - 1
    cum_return = port_val.iloc[-1]/port_val[0]-1			
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(samples_per_year)*(avg_daily_ret-daily_rf)/std_daily_ret  		 

    return cum_return, avg_daily_ret, std_daily_ret, sharpe_ratio	
