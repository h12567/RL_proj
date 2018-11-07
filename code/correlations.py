import os
import numpy as np
import pandas as pd
import itertools

stock_dir = "../data/stock/"

filenames = []
for file in os.listdir(stock_dir):
    filename = os.fsdecode(file)
    filenames.append(filename[:-4])  # remove .csv

n = len(filenames)
r = 6

list.sort(filenames)

stock_prices = []
for filename in filenames:
    stock_csv = pd.read_csv(stock_dir + filename + ".csv")
    stock_price = list(stock_csv.Open)
    stock_prices.append(stock_price)

cov_mat = np.cov(stock_prices)


i = 0
min_variance = 100000000
best_result = []
for combination in itertools.combinations(range(len(filenames)), r):
    stocks = list(combination)

    cov_sub = cov_mat[np.ix_(stocks, stocks)]
    inv_cov = np.linalg.inv(cov_sub)

    mat_sum = inv_cov.sum()
    weight = inv_cov.sum(axis=1) / mat_sum
    variance = 1 / mat_sum

    result = str(np.array(filenames)[np.ix_(stocks)]) + "," + str(weight) + ',' + str(variance) + "\n"

    if min_variance > variance and (weight > 0).all():  # short selling not allowed
        min_variance = variance
        best_result = result

    i += 1
    if i % 20000 == 0:
        print(best_result)

print(best_result)