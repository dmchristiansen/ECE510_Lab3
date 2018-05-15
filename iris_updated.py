"""
 ECE 510 - Lab 3
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

# globals
data_file = 'iris.data.txt'


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

    # convert floats to u4.4 format
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
    def __init__(self, h1_size, h2_size, eta, samples, labels, epochs=1):

        WEIGHT_FRAC_BITS = 12

        # intialize weights for input -> hidden layer 1
        self.w_i1 = np.int16((np.random.random((h1_size, 4)) - 0.5) * (2 ** (WEIGHT_FRAC_BITS - 3)))
        # initialize weights for hidden layer 1 -> hidden layer 2
        self.w_12 = np.int16((np.random.random((h2_size, h1_size)) - 0.5) * (2 ** (WEIGHT_FRAC_BITS - 3)))
        # initialize weights for hidden layer 2 -> output
        self.w_2o = np.int16((np.random.random((3, h2_size)) - 0.5) * (2 ** (WEIGHT_FRAC_BITS - 3)))

        self.hidden1_size = h1_size
        self.hidden2_size = h2_size

        self.s16_min = (-(2 ** 15))
        self.s16_max = ((2 ** 15) - 1)
        self.s32_min = (-(2 ** 31))
        self.s32_max = ((2 ** 31) - 1)
        self.s40_min = (-(2 ** 39))
        self.s40_max = ((2 ** 39) - 1)

        self.LUT = np.loadtxt('sigmoid_LUT.txt', dtype=np.uint16)
        self.eta = eta
        self.samples = samples
        self.labels = labels
        self.epochs = epochs


    def fp_dot(self, a, b):
        accumulator = np.int64(0)   # 40 bit accumulator, plus room for overflow
        product = np.int64(0)       # 32 bit product, plus room for overflow
        for i in range(a.shape[0]):
            product = np.int64(a[i]) * np.int64(b[i])
            if product > self.s32_max:
                product = np.int64(self.s32_max)
            elif product < self.s32_min:
                product = np.int64(self.s32_min)
            accumulator += product

        if accumulator > self.s40_max:
            accumulator = np.int64(self.s40_max)
        elif accumulator < self.s40_min:
            accumulator = np.int64(self.s40_min)

        return accumulator

    def fp_mat_mul(self, w, x):
        if(w.shape[1] != x.shape[0]):
            print("Mismatched matrix dimensions")
            return -1

        result = np.asarray([self.fp_dot(w[row], x) for row in range(w.shape[0])])
        return result

    """
    Fixed-point sigmoid approximation
    Uses 256 element LUT, plus interpolation
    """
    def sigmoid(self, x):
        bitmask = np.int16(255)

        # format is s3.12, need to offset by
        index = np.bitwise_and(np.right_shift(x, 8) + 127, bitmask)
        result = self.LUT[index]

        fractional = np.bitwise_and(x, bitmask)
        diff = self.LUT[index+1] - self.LUT[index]
        result += np.right_shift(np.uint16(diff * fractional), 8, dtype=np.uint16)

        return result

    """
    Takes array of values, replaces max value with 1, all others with 0
    Returns np.uint8 array
    """
    def one_hot(self, y):
        prediction = np.zeros(y.shape)
        prediction[np.argmax(y)] = 1
        return prediction

    """
    Forward propagation routine
    Currently not using a bias input
    """
    def fit(self, samples, labels):
        output_ = []
        label_ = []
        accuracy_on_test = []
        accuracy_on_train = []
        no_epochs = []
        sample_no = 0
        
        for epoch in range(self.epochs):
            for i in range(0,len(self.samples),1):
                sample_no += 1
                x = self.samples[i]
                y = self.labels[i]
                
                # Feed Forword pass
                # reformat x from u4.4 to u4.12, multiply with weights
                hidden_1 = self.fp_mat_mul(self.w_i1, (np.int16(x)*(2**12)))
                # result is in s15.24 format, reformat to s3.12
                hidden_1 = np.int16(np.right_shift(hidden_1, 12))
                # apply sigmoid function, which returns format u4.12
                hidden_1 = self.sigmoid(hidden_1)
           
                # multiply hidden layer 1 output with hidden layer 2's weights
                # u4.12 (h1 output) * s3.12 (weights) = s15.24
                hidden_2 = self.fp_mat_mul(self.w_12, hidden_1)
                # reformat from s15.24 to s3.12
                hidden_2 = np.int16(np.right_shift(hidden_2, 12))
                hidden_2 = self.sigmoid(hidden_2)
              
                # multiply hidden layer 2 output with output layer's weights
                # u4.12 (h2 output) * s3.12 (weights) = s15.24
                output = self.fp_mat_mul(self.w_2o, hidden_2)
                # reformat from s15.24 to s3.12
                output = np.int16(np.right_shift(output, 12))
                output = self.sigmoid(output)
                
                label = self.one_hot(output)
                output_.append(output)
                label_.append(label)
                '''
                # Backpropagation
                # calculate output layer error
                output_delta = (output - y) 
                output_delta = output_delta * output * (1 - output)
                output_delta = np.dot(hidden_2.T, output_delta)

                # calculate hidden layer error
                hidden_2_delta = np.dot(self.w_2o, output_delta.T)
                hidden_2_delta = hidden_2_delta * (hidden_2 * (1 - hidden_2)).T
                hidden_2_delta = np.dot(hidden_1.T, hidden_2_delta.T)
               
                hidden_1_delta = np.dot(self.w_12, hidden_2_delta)
                hidden_1_delta = hidden_1_delta * (hidden_1 * (1 - hidden_1)).T
                hidden_1_delta = np.dot(x.T, hidden_1_delta.T)
                
                # weights update
                self.w_2o -= self.eta * output_delta
                self.w_12 -= self.eta * hidden_2_delta
                self.w_i1 -= self.eta * hidden_1_delta
                '''

            return output_,label_
    
if __name__ == '__main__':

    x, y = load_dataset()
    y_ = y.reshape((len(x), 1))
    enc = OneHotEncoder()
    enc.fit(y_)
    labels_ = enc.transform(y_).toarray()   # one hot encoding of y
    
    samples = x[0:10]
    labels = labels_[0:10]
    eta = 0
    
    
    MLP = Network(3, 3, eta, samples, labels)
 
    outputs_, labels_ =  MLP.fit(samples, labels)
