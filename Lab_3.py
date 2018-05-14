"""
 ECE 510 - Lab 3


"""

import numpy as np

# globals
data_file = 'iris.data'


"""
 Loads dataset and converts to proper format

 Attributes are read in as floats, from 0 - 7.9,
 then converted to u4.4 format.

 Classes are read in as strings and converted to ints as follows:
 'Iris-setosa'     = 0
 'Iris-versicolor' = 1
 'Iris-virginica'  = 2

"""
def load_dataset():
    F_BITS = 4

    # load in data from csv
    x = np.loadtxt(data_file, dtype='float', delimiter=',', usecols=(0, 1, 2, 3))
    y = np.loadtxt(data_file, dtype='S30', delimiter=',', usecols=(4))

    # convert floats to u8 format
    x = np.uint8(x * (2 ** F_BITS) + 0.5)

    # convert classes from string to int
    y_int = np.zeros(y.shape[0], dtype=np.uint8)
    y = y.astype(str)
    for i in range(y.shape[0]):
        if(np.core.defchararray.equal(y[i], 'Iris-setosa')):
            y_int[i] = 0
        elif(np.core.defchararray.equal(y[i], 'Iris-versicolor')):
            y_int[i] = 1
        elif(np.core.defchararray.equal(y[i], 'Iris-virginica')):
            y_int[i] = 2

    return x, y_int

"""
 MLP class
"""
class Network:

    """
    Initializes weights in range (-0.0625, 0.0625), in s3.12 format
    Currently no bias input
    """
    def __init__(self, h1_size, h2_size):

        WEIGHT_FRAC_BITS = 12

        # intialize weights for input -> hidden layer 1
        self.w_i1 = np.int16((np.random.random((h1_size, 4)) - 0.5) * (2 ** (WEIGHT_FRAC_BITS - 3)))
        # initialize weights for hidden layer 1 -> hidden layer 2
        self.w_12 = np.int16((np.random.random((h2_size, h1_size)) - 0.5) * (2 ** (WEIGHT_FRAC_BITS - 3)))
        # initialize weights for hidden layer 2 -> output
        self.w_2o = np.int16((np.random.random((3, h2_size)) - 0.5) * (2 ** (WEIGHT_FRAC_BITS - 3)))

        self.hidden1_size = h1_size
        self.hidden2_size = h2_size


    """
    Fixed-point sigmoid approximation
    Uses 256 element LUT, plus interpolation
    """
    def sigmoid(self, x):
        # note: write this function
        return x

    """
    Takes array of values, replaces max value with 1, all others with 0
    Returns np.uint8 array
    """
    def one_hot(self, y):
        # note: write this function
        return y

    """
    Forward propagation routine
    Currently not using a bias input
    """
    def forward_propagate(self, x):
        # note: write this function
        return x


if __name__ == '__main__':

    x, y = load_dataset()
    MLP = Network(3, 3)

















