import datetime as dt
import numpy as np
import pandas as pd
import copy
import util as ut  	
#import marketsimcode as ms
import matplotlib.pyplot as plt
import StrategyLearner as sl
import ManualStrategy as ms
from marketsimcode import compute_portvals, get_portfolio_stats

def author():
    return 'axiao31'

def experiment1():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)
    symbols = 'JPM'
    prices= ut.get_data([symbols], dates)
    prices = prices.fillna(method='ffill')
    prices = prices.fillna(method='bfill')

    #benchmark 
    bm = pd.DataFrame(index=prices.index)
    bm[symbols] = 0
    #print(bm)
    #bm = pd.DataFrame(columns=['Order', "Shares"])
    bm.loc[bm.index.min(), symbols] = 1000
    bm.loc[bm.index.max(), symbols] = -1000
    portval = compute_portvals(bm, sd, ed, start_val=100000, commission=9.95, impact=0.005)
    bm_portval = portval / portval.iloc[0] 

    cr, addr, sddr, sr = get_portfolio_stats(portval)
    print("Benchmark:")
    print("Cumulative Return: ", str(cr))
    print("Average Daily Return: ", str(addr))
    print("Standard Deviation of Daily Return: ", str(sddr))
    print("Sharpe Ratio: ", str(sr))
    print("")

    #manual strategy
    manual = ms.ManualStrategy().testPolicy(symbols, sd, ed, sv=100000)
    portval = compute_portvals(manual, sd, ed, start_val=100000, commission=0, impact=0)
    ms_portval = portval / portval.iloc[0]
    
    cr, addr, sddr, sr = get_portfolio_stats(portval)
    print("Manual Strategy:")
    print("Cumulative Return: ", str(cr))
    print("Average Daily Return: ", str(addr))
    print("Standard Deviation of Daily Return: ", str(sddr))
    print("Sharpe Ratio: ", str(sr))
    print("")

    #strategy learner
    learner = sl.StrategyLearner(verbose=False, impact=0)
    learner.add_evidence(symbols, sd, ed, sv=100000)
    sl_trades = learner.testPolicy(symbols, sd, ed, sv=100000)
    sl_trades = pd.DataFrame(sl_trades, columns=[symbols])
    portval = compute_portvals(sl_trades, sd, ed, start_val=100000, commission=9.95, impact=0.005)
    sl_portval = portval/portval.iloc[0]

    cr, addr, sddr, sr = get_portfolio_stats(portval)
    print("Strategy Learner:")
    print("Cumulative Return: ", str(cr))
    print("Average Daily Return: ", str(addr))
    print("Standard Deviation of Daily Return: ", str(sddr))
    print("Sharpe Ratio: ", str(sr))
    print("")

    #plot
    fig, ax = plt.subplots()
    ax = bm_portval.plot( label="Benchmark", color = "purple")
    ms_portval.plot(ax=ax, label="Manual Strategy", color = "red")
    sl_portval.plot(ax=ax, label="Strategy Learner")
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title("Experiment 1")
    plt.savefig("images/experiment1.png")





