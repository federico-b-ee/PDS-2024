import numpy as np
import sys
import os

sys.path.append(os.path.abspath("."))
from pds_utils import adc, plotter, sine

fs = N = 10000
f = 100
t, sr = sine.wave(sampling_freq=fs, samples=N, amplitude=2, dc_level=0, frequency=f)


sq1, q1 = adc.adc(sr, bits=2, vcc=2, vee=-2)
sq2, q2 = adc.adc(sr, bits=3, vcc=2, vee=-2)
sq3, q3 = adc.adc(sr, bits=4, vcc=2, vee=-2)
sq4, q4 = adc.adc(sr, bits=8, vcc=2, vee=-2)

e1 = sq1 - sr
e2 = sq2 - sr
e3 = sq3 - sr
e4 = sq4 - sr

plotter.multiple_cmp_adc(
    [([sq1, e1, sr], 2), ([sq2, e2, sr], 3), ([sq3, e3, sr], 4), ([sq4, e4, sr], 8)],
    t,
    f,
)

variance_error1 = np.var(e1)
variance_errorq1 = q1**2 / 12.0

print(f"Varianza del error: {variance_error1}")
print(f"Varianza del error: {variance_errorq1}")
