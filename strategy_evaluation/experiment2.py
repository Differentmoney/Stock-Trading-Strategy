import datetime as dt
import numpy as np
import pandas as pd
import copy
import util as ut  	
import marketsimcode as ms
import matplotlib.pyplot as plt
import StrategyLearner as sl


def author():
    return 'axiao31'

def show_stats(portval, impact):
    cr, addr, sddr, sr = ms.get_portfolio_stats(portval)
    print("Learner with impact:" + str(impact))
    print("Cumulative Return: ", str(cr))
    print("Average Daily Return: ", str(addr))
    print("Standard Deviation of Daily Return: ", str(sddr))
    print("Sharpe Ratio: ", str(sr))
    print("")

def experiment2():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)
    symbols = 'JPM'
    
    impacts = [0.0005, 0.005, 0.05]
    portval = []

    for i in impacts:
        learner = sl.StrategyLearner(verbose=False, impact=i)
        learner.add_evidence(symbols, sd, ed, sv=100000)
        sl_trades = learner.testPolicy(symbols, sd, ed, sv=100000)
        df = pd.DataFrame(sl_trades, columns = [symbols])
        # print(type(df))
        # print(df)
        temp = ms.compute_portvals(df, sd, ed, start_val=100000, commission=9.95, impact=i)
        temp = temp/temp.iloc[0]
        portval.append(temp)
        show_stats(temp, i)
    
    #plot 
    fig, ax = plt.subplots()
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Portfolio Value")
    ax.set_title("Cumulative Return vs. Impact")
    ax = portval[0].plot(label="0.0005")
    portval[1].plot(label="0.005")
    portval[2].plot(label="0.05")
    ax.legend(loc='best')
    plt.savefig("images/experiment2.png")

    

