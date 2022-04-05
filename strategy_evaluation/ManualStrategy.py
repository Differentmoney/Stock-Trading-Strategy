  	   		  	  			  		 			     			  	 
import datetime as dt  		  	   		  	  			  		 			     			  	 
import random  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
import util as ut  		  	   		  	  			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
from indicators import *   		 
from marketsimcode import compute_portvals, get_portfolio_stats 	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class ManualStrategy(object):  		  	   		  	  			  		 			     			  	 		  		 			     			  	 
    # constructor  		  	   		  	  			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Constructor method  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        self.verbose = verbose  		  	   		  	  			  		 			     			  	 
        self.impact = impact  		  	   		  	  			  		 			     			  	 
        self.commission = commission  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		  	  			  		 			     			  	 
    def add_evidence(  		  	   		  	  			  		 			     			  	 
        self,  		  	   		  	  			  		 			     			  	 
        symbol="IBM",  		  	   		  	  			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		  	  			  		 			     			  	 
        sv=10000,  		  	   		  	  			  		 			     			  	 
    ):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		  	  			  		 			     			  	 
        :type symbol: str  		  	   		  	  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
        :type sd: datetime  		  	   		  	  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
        :type ed: datetime  		  	   		  	  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
        :type sv: int  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        # add your code to do learning here  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        # example usage of the old backward compatible util function  		  	   		  	  			  		 			     			  	 
        syms = [symbol]  		  	   		  	  			  		 			     			  	 
        dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			     			  	 
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		  	  			  		 			     			  	 
        prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  			  		 			     			  	 
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  	  			  		 			     			  	 
        if self.verbose:  		  	   		  	  			  		 			     			  	 
            print(prices)  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        # example use with new colname  		  	   		  	  			  		 			     			  	 
        volume_all = ut.get_data(  		  	   		  	  			  		 			     			  	 
            syms, dates, colname="Volume"  		  	   		  	  			  		 			     			  	 
        )  # automatically adds SPY  		  	   		  	  			  		 			     			  	 
        volume = volume_all[syms]  # only portfolio symbols  		  	   		  	  			  		 			     			  	 
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later  		  	   		  	  			  		 			     			  	 
        if self.verbose:  		  	   		  	  			  		 			     			  	 
            print(volume)  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		  	  			  		 			     			  	 
    def testPolicy(  		  	   		  	  			  		 			     			  	 
        self,  		  	   		  	  			  		 			     			  	 
        symbol="IBM",  		  	   		  	  			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		  	  			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		  	  			  		 			     			  	 
        sv=10000,  		  	   		  	  			  		 			     			  	 
    ):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  	  			  		 			     			  	 
        :type symbol: str  		  	   		  	  			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
        :type sd: datetime  		  	   		  	  			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
        :type ed: datetime  		  	   		  	  			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			     			  	 
        :type sv: int  		  	   		  	  			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 

        # Get the data  		  	   		  	  			  		 			     			  	 
        dates = pd.date_range(sd, ed)   
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbols

        lookback = 30
        _, _, bb  = get_bb_bands(prices, lookback)
        momentum = get_momentum(prices, lookback)
        sma_price = get_sma(prices, lookback)

        trades = pd.DataFrame(columns = ['Order', 'Shares'], index = prices.index)
        signal = 0

        for i in range(prices.shape[0]):
            index = prices.index[i]
            if signal == 0:
                if momentum.loc[index] < -0.1 or bb.loc[index] < 0.2 or sma_price.loc[index] < 0.6:
                    trades.loc[index]= ['BUY', 1000]
                    signal = -1
                elif  momentum.loc[index] > 0.1  or bb.loc[index] > 0.7 or sma_price.loc[index] < 1.0:
                    trades.loc[index]= ['SELL', 1000]
                    signal = 1
            elif signal == 1:
                if momentum.loc[index] > 0.2 or bb.loc[index] > 0.8 or  sma_price.loc[index] > 1.4:
                    trades.loc[index]= ['SELL', 2000]
                    signal = -1
            elif signal == -1:
                if momentum.loc[index] < -0.1 or bb.loc[index] < 0.3 or  sma_price.loc[index] < 0.6:
                    trades.loc[index]= ['BUY', 2000]
                    signal = 1
            
        trades['Shares'] = np.where(trades['Order'] == 'BUY', trades['Shares'], -trades['Shares'])
        trades = trades.loc[:,['Shares']]
        trades.columns = [symbol]
        trades.fillna(0, inplace=True)
      
        
        return trades
        
    def evaluate( sd, ed, title):
        sym = 'JPM'
        sv = 100000
        lookback = 30

        dates = pd.date_range(sd, ed)
        prices= ut.get_data([sym], dates)
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')

        #benchmark 
        bm = pd.DataFrame(index=prices.index)
        bm[sym] = 0
        #print(bm)
        #bm = pd.DataFrame(columns=['Order', "Shares"])
        bm.loc[bm.index.min(), sym] = 1000
        bm.loc[bm.index.max(), sym] = -1000
        portval = compute_portvals(bm, sd, ed, start_val=100000, commission=9.95, impact=0.005)
        bm_portval = portval / portval.iloc[0] 

        cr, addr, sddr, sr = get_portfolio_stats(portval)
        print("Benchmark: ", title)
        print("Cumulative Return: ", str(cr))
        print("Average Daily Return: ", str(addr))
        print("Standard Deviation of Daily Return: ", str(sddr))
        print("Sharpe Ratio: ", str(sr))
        print("")

        # manual strategy
        manual = ManualStrategy().testPolicy(sym, sd, ed, sv=100000)
        portval = compute_portvals(manual, sd, ed, start_val=100000, commission=0, impact=0)
        ms_portval = portval / portval.iloc[0]
        
        cr, addr, sddr, sr = get_portfolio_stats(portval)
        print("Manual Strategy: ", title)
        print("Cumulative Return: ", str(cr))
        print("Average Daily Return: ", str(addr))
        print("Standard Deviation of Daily Return: ", str(sddr))
        print("Sharpe Ratio: ", str(sr))
        print("")
  		  	   		  	  			  		 			     			  	 
        # plot the two curves
        ax = plt.subplots()
        ax = bm_portval.plot(label="Benchmark", color = "green")
        ms_portval.plot(ax=ax, label="Manual Strategy", color = "red")
        # label short and long positions
        for i, signal in manual.iterrows():
            amount = manual.loc[i, sym]
            if amount < 0:
                plt.axvline(x=i, color='blue', linestyle='--')
            elif amount > 0:
                plt.axvline(x=i, color='black', linestyle='--')

        plt.legend(loc='upper left')
        plt.xlabel("Date")
        plt.ylabel("Normalized Portfolio Value")
        plt.title("Manual Strategy vs Benchmark - " + title)
        plt.savefig('ms_'+title+'.png')     

if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		  	  			  		 			     			  	 
