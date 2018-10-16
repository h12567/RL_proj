import pandas as pd
import datetime
import numpy as np
# from pandas import *
# codes = ['AAPL', 'BABA', 'V']
# codes = ['AAPL', 'GOOG', 'BABA', 'V']
codes = ['XOM', 'AAPL', 'BP', 'GE', 'INTC', 'AMGN', 'JPM', 'CAT', 'PFE',
       'CVX', 'UNH', 'MCD', 'D', 'WMT', 'PEP', 'PG', 'WFC', 'CSCO',
       'GOOG', 'DIS', 'HD', 'C', 'BAC', 'AMZN', 'VZ', 'CELG', 'MRK',
       'MMM', 'KO', 'V', 'MA', 'BA', 'JNJ', 'MSFT', 'T', 'ORCL', 'FB',
       'CMCSA', 'MO', 'PCLN']
final = 1
for code in codes:
  TOTAL_ROWS_NASDAQ = 50000
  USE_COLS = [0,1,2,3,4,5,6]
  nasdaq = pd.read_csv("./raw/" + code + ".csv", skiprows=1, nrows=TOTAL_ROWS_NASDAQ, usecols=USE_COLS, header=None)
  nasdaq = nasdaq.values
  # newNasdaq = np.array(shape=(nasdaq.shape[0], nasdaq.shape[1] + 1), dtype=object)
  newNasdaq = []
  for i in range(nasdaq.shape[0]):
    day = datetime.datetime.strptime(nasdaq[i, 0], '%Y-%m-%d')
    nasdaq[i, 0] = day.strftime('%B %d, %Y')
    newNasdaq.append(np.append(nasdaq[i], [code], axis=0))

  if(code == codes[0]):
    final = newNasdaq
  else:
    final = np.append(np.array(final), np.array(newNasdaq), axis=0)

final = np.append(np.array([['time', 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'code']]), final, axis=0)

final = pd.DataFrame(final)
# names(final) <- NULL
print(final)
final.to_csv('nasdaq.csv', header=False)