import numpy as np
from numpy.polynomial import Polynomial


coef = [-203.606, 1523.29, -3196.413, 2474.455, 3.855326]
# coef.reverse()
water_cp = Polynomial(coef)
water_integ = water_cp.integ().convert()

t1 = 298.15
t2 = 398.15

print((water_integ(t2 / 1000) - water_integ(t1 / 1000)) / (t2 - t1))
print(water_cp)
