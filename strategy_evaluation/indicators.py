import datetime as dt
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

def author():
    return 'axiao31'

def get_bb_bands(df, lookback):
    bb = df.copy()
    avg = df.rolling(lookback).mean()
    std = df.rolling(lookback).std()
    bb = (df - avg)/(std*2)
    upper = avg + (std*2)
    lower = avg - (std*2)
    bb = (df - lower)/ (upper - lower)
    
    #Plot the Bollinger Bands
    # plt.plot(avg,label='SMA')
    # plt.plot(upper, label='Upper Band')
    # plt.plot(lower, label='Lower Band')
    # plt.legend(loc='best')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.title('Bollinger Bands')
    # plt.savefig('BollingerBands.png')
    # plt.clf()
    
    return upper, lower, bb

def get_sma(df,lookback1):
    sma = df.rolling(window=lookback1, center=False).mean()
    #sma2 = df.rolling(window=lookback2, center=False).mean()

    sma = df.divide(sma, axis=0)
    # Plot the SMA
    # plt.plot(df, label='Price')
    # plt.plot(sma, label='SMA(10)')
    # plt.plot(sma2, label='SMA(50)')
    # plt.legend(loc='best')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.title('Simple Moving Average')
    # plt.savefig('SMA.png')
    # plt.clf()
    return sma

def get_momentum(df, lookback):
    mm = df/df.shift(lookback) - 1
    #plot the momentum
    # plt.plot(mm, label='Momentum')
    # plt.plot(df, label='Price')
    # plt.legend(loc='best')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.title('Momentum')
    # plt.savefig('Momentum.png')
    # plt.clf()
    return mm

def macd(df, lookback):
    # Based on https://www.alpharithms.com/calculate-macd-python-272222/
    s = df.ewm(span=12, adjust=False).mean()
    l = df.ewm(span=26, adjust=False).mean()
    macd = s - l
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    # Plot the MACD
    # plt.plot(macd, label='MACD')
    # plt.plot(macd_signal, label='MACD Signal')
    # plt.legend(loc='best')
    # plt.xlabel('Date')
    # plt.ylabel('MACD Value')
    # plt.title('MACD and MACD Signal')
    # plt.savefig('MACD.png')
    # plt.clf()

    # plt.plot(df, label='Price')
    # plt.plot(s, label='EMA(12)')
    # plt.plot(l, label='EMA(26)')
    # plt.legend(loc='best')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.title('EMA(12) and EMA(26)')
    # plt.savefig('EMA.png')
    # plt.clf()

    return macd, macd_signal

def cci(df, lookback):
    # From https://blog.quantinsti.com/build-technical-indicators-in-python/
    # CCI = (Typical price – MA of Typical price) / (0.015 * mean deviation of Typical price)
    cci = (df - df.rolling(window=lookback).mean()) / (df.rolling(window=lookback).std() * 0.015)
    # Plot the CCI
    # plt.plot(cci, label='CCI')
    # plt.legend(loc='best')
    # plt.xlabel('Date')
    # plt.ylabel('CCI Value')
    # plt.title('Commodity Channel Index')
    # plt.savefig('CCI.png')
    # plt.clf()

    return cci



