import os
import numpy as np
import pandas as pd


def valid(stockfilename):
    return stockfilename.endswith(".csv") and stockfilename != "CommissionFee.csv"


errfile = open("./error.txt", "w+")
covfile = open("./cov.txt", "w+")

filenames = []
for file in os.listdir("../data/"):
    filename = os.fsdecode(file)
    if valid(file):
        filenames.append(filename)

n = len(filenames)
for i in range(n):
    filename1 = filenames[i]
    stock_csv1 = pd.read_csv(filename1)
    stock_price1 = stock_csv1.Open

    for j in range(i, n):
        filename2 = filenames[j]
        stock_csv2 = pd.read_csv(filename2)
        stock_price2 = stock_csv2.Open

        try:
            cov = np.corrcoef(stock_price1, stock_price2)
            print(filename1[:-4], filename2[:-4])
            covfile.write(filename1[:-4] + "," + filename2[:-4] + "," + str(cov[0][1]) + "\n")
        except:
            errfile.write("Not working for " + filename1[:-4] + " and " + filename2[:-4] + "\n")

errfile.close()
covfile.close()