import matplotlib.pyplot as plt
import numpy as np
import pywt
import math

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
        t = math.sqrt(2*math.log(x.shape[0])) * np.median(abs(coeffs[-1])) / 0.6745
        for coe in coeffs:
            new_coeffs.append(pywt.threshold(coe, t, "soft"))
        new_x = pywt.waverec(new_coeffs, db1)
        return new_x
        # plt.plot(self.index, new_open)
        # plt.show()
