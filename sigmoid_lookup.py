# script to generate sigmoid function lookup table

import numpy as np

LUT = np.uint16((1/(1 + np.exp(-1 * (np.arange(-128, 128) / (2**5))))) * (2**12) )

print(LUT)
np.savetxt("sigmoid_LUT.txt", LUT, delimiter=",", fmt='%5u')
