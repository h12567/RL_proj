import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import math
import os

# Note: need to install pywt and PyWavelets
class denoise_data():

    def __init__(self, level=1):
        # level: the level that the multilevel wavelet transform to decompose
        self.level = level

    def process(self, x):
        # x: 1D-array
        db1 = pywt.Wavelet('db1')
        coeffs = pywt.wavedec(x, db1, level=self.level)
        new_coeffs = list()
        d = coeffs[-1]
        d = np.nan_to_num(d)
        t = math.sqrt(2*math.log(x.shape[0])) * np.median(abs(d)) / 0.6745
        for coe in coeffs:
            new_coeffs.append(pywt.threshold(coe, t, "soft"))
        new_x = pywt.waverec(new_coeffs, db1)
        if len(new_x) != len(x): # sometimes differ by 1
            new_x = new_x[:len(x)]
        return new_x


root_path = "../data"
def main():
    data = [os.path.join(root_path, fi) for fi in os.listdir(root_path)]

    for data_path in data:
        process(data_path)

    # process(os.path.join(root_path, "PTR.csv"))



def process(data_path):
    wavelet_processor = denoise_data()
    file_name = data_path.split(os.sep)[-1]
    if file_name == "CommissionFee.csv" or file_name == "new_data":
        return
    stock_df = pd.read_csv(data_path, sep=",")
    open = stock_df["Open"].values
    close = stock_df["Close"].values
    high = stock_df["High"].values
    low = stock_df["Low"].values
    volume = stock_df["Volume"].values
    new_open = wavelet_processor.process(open)
    new_close = wavelet_processor.process(close)
    new_high = wavelet_processor.process(high)
    new_low = wavelet_processor.process(low)
    result_dict = {'Date': stock_df["Date"].values, 'Open': open, 'Close': close,
                   "High": high, "Low": low, "Adj Close": stock_df["Adj Close"].values,
                   "Volume": volume, "Open_after_wavelet": new_open, "Close_after_wavelet": new_close,
                   "High_after_wavelet": new_high, "Low_after_wavelet": new_low}
    result = pd.DataFrame(data=result_dict)
    result.to_csv(os.path.join("../data/new_data", file_name), index=False,
                  columns=["Date", "Open", "Close", "High", "Low", "Adj Close", "Volume",
                           "Open_after_wavelet", "Close_after_wavelet", "High_after_wavelet",
                           "Low_after_wavelet"])


main()
