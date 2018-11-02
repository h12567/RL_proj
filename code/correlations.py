import os
import io
import numpy as np
import pandas as pd
import itertools

stock_dir = "../data/stock/"

filenames = []
for file in os.listdir(stock_dir):
    filename = os.fsdecode(file)
    filenames.append(filename)

n = len(filenames)

list.sort(filenames)

stock_prices = []
for filename in filenames:
    stock_csv = pd.read_csv(stock_dir + filename)
    stock_price = list(stock_csv.Open)
    stock_prices.append(stock_price)

cov_mat = np.cov(stock_prices)
inv_cov_mat = np.linalg.inv(cov_mat)

i = 0
r = 6
with io.open('../data/correlation/mvp' + str(r) + '.txt', 'w') as file:
    for combination in itertools.combinations(range(len(filenames)), r):
        stocks = list(combination)

        cov_sub = cov_mat[np.ix_(stocks, stocks)]
        inv_cov = np.linalg.inv(cov_sub)
        variance = 1 / inv_cov.sum()

        result = ''.join(filenames[i][:-4] + "," for i in stocks) + str(variance) + "\n"
        file.write(result)

        i += 1
        if i % 10000 == 0:
            print(i)
