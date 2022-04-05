 		  	   		  	  			  		 			     			  	 
import datetime as dt  		  	   		  	  			  		 			     			  	 
import random  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
import util as ut  	
import numpy as np
from learners import BagLearner as bl
from learners import RTLearner as rt
from indicators import *	  	   		  	  			  		 			     			  	 
	  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
class StrategyLearner(object):  		  	   		  	  			  		 			     			  	 	
		  	  			  		 			     			  	 
    # constructor  		  	   		  	  			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        Constructor method  		  	   		  	  			  		 			     			  	 
        """  		  	   		  	  			  		 			     			  	 
        self.verbose = verbose  		  	   		  	  			  		 			     			  	 
        self.impact = impact  		  	   		  	  			  		 			     			  	 
        self.commission = commission  	
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size":5}, bags=20, boost=False, verbose=False)	  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
        sym=[symbol]
        dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			     			  	 
        prices_all = ut.get_data(sym, dates)  # automatically adds SPY  		  	   		  	  			  		 			     			  	 
        prices = prices_all[symbol]  # only portfolio symbols  		                                                                                                                                                                                	   		  	  			  		 			     			  	 
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices/ prices.iloc[0]
        	
   	   		  	  			  		 			     			  	 
        if self.verbose:  		  	   		  	  			  		 			     			  	 
            print(prices)  
        lookback = 20		  	   		  	  			  		 			     			  	 
  			  	  			  		 			     			  	 
        _, _, bb  = get_bb_bands(prices, lookback)
        momentum = get_momentum(prices, lookback)
        sma_price = get_sma(prices, lookback)
        df = pd.concat((bb, sma_price, momentum), axis=1)
        df.columns = ['bb', 'sma_price', 'momentum']
        df = df.fillna(0)

        period = 10
        x_train = df[:-period].values
        y_train = np.zeros(x_train.shape[0])

        buy_trigger = 0.02 + self.impact
        sell_trigger = -0.02 - self.impact

        #print("Prices: ", prices)
        for i in range(prices.shape[0] - period):
            threshold = (prices.loc[prices.index[i+10]]/prices.loc[prices.index[i]])-1.0
            if threshold > buy_trigger:
                y_train[i] = 1
            elif threshold < sell_trigger:
                y_train[i] = -1
            else:
                y_train[i] = 0
        
        # add data to learner
        self.learner.add_evidence(x_train, y_train)
        		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
  		  	   		  	  			  		 			     			  	 
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbols
        if self.verbose:
            print(prices)
        lookback = 30
        _, _, bb  = get_bb_bands(prices, lookback)
        momentum = get_momentum(prices, lookback)
        sma_price = get_sma(prices, lookback)

        # get the data for the learner
        x_test = pd.concat((bb, sma_price, momentum), axis=1)
        x_test.columns = ['bb', 'sma_price', 'momentum']
        x_test = x_test.fillna(0)
        x_test = x_test.values
        y = self.learner.query(x_test)
        #y_test = pd.DataFrame(y.T, index=prices.index, columns=[symbol])
        y_test = y
        trades = prices.copy()
        trades[:] = 0
        
        signal = 0

        # print("Price shape",prices.shape)
        # print("Y value",y[0][0])
        # print("Trades", trades.loc['2008-01-02 00:00:00'])

        for i in range(prices.shape[0]-1):
            index = prices.index[i]
            if signal == 0 and y_test[0][i]>0:
                trades.loc[index] = 1000
                signal = 1
            elif signal == 0  and y_test[0][i]<0:
                trades.loc[index] = -1000
                signal = -1
            elif signal == 1 and y_test[0][i]<0:
                trades.loc[index] = -2000
                signal = -1
            elif signal == -1 and y_test[0][i]>0:
                trades.loc[index] = 2000
                signal = 1
                
        return trades

  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    print("One does not simply think up a strategy")  		  	   		  	  			  		 			     			  	 
