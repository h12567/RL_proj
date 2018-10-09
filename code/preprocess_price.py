import os
from os import listdir
from os.path import isfile, join
import numpy as np
import json
import datetime
import pandas as pd

def normalize(data):
    max_data = np.max(data, axis = 0)
    min_data = np.min(data, axis = 0)
    data = np.array(list(map(lambda x: (2* x - (max_data + min_data)) / (max_data - min_data), data)))
    return data
  
def preprocess_price(stock):
    date1 = '2013-01-01'
    date2 = '2016-12-31'
    start = datetime.datetime.strptime(date1, '%Y-%m-%d')
    end = datetime.datetime.strptime(date2, '%Y-%m-%d')
    step = datetime.timedelta(days=1)
    prices = []
    TOTAL_ROWS_NASDAQ = 50000
    USE_COLS = [0, 4]

    nasdaq = pd.read_csv("../data/" + stock + ".csv", skiprows=1, nrows=TOTAL_ROWS_NASDAQ, usecols=USE_COLS, header=None)
    nasdaq = nasdaq.values

    for i in range(nasdaq.shape[0]):
        current_date = nasdaq[i, 0]
        current_year = current_date.split("-")[0]
        price = nasdaq[i, 1]
        if (current_year =='2013' or current_year == '2014' or current_year == '2015' or current_year == '2016'):
            while(current_date != str(start.date())):
                prices.append(np.nan)
                start += step
            prices.append(price)
            start += step
    while(start <= end):
        prices.append(np.nan)
        start += step
    prices = np.flip(np.array(prices), axis = 0)
    prices = pd.DataFrame(prices)
    prices = prices.fillna(method='bfill')
    prices = prices.fillna(method='ffill')
    prices = prices.values.flatten()
    # prices = normalize(prices)
    return np.array(prices).tolist()

