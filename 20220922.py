import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import trapezoid

# A + B -> C + D
# A: MBP
# B: n-butanol
# C: DPB
# D: H2O


# TODO: test Pint

CA0 = 0.2 # lbmol / ft3
k = 1.2 # ft3 / lbmol * h
FA0 = 1 * 60 # lbmol / h
    

def calculate_pfr_volume(Xi, X, n=64):
    x = np.linspace(Xi, X, n)
    y = 1 / ((1 - x) * (5 - x))
    return  FA0 / (k * CA0 * CA0) * trapezoid(y, x)


print(calculate_pfr_volume(0, 0.85))
