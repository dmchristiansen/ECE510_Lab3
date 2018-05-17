from numpy import exp, array, random, dot
import numpy as np
from sklearn.datasets import load_iris

class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        # Number of nodes in hidden layers
        l2 = 3     # layer 2
        l3 = 3     # Layer 3
        
        # Learning rate
        self.eta = 0.1

        # assign random weights
        self.weights1 = np.random.uniform(low = -0.05, high= +0.05, size= ((4, l2)))
        self.weights2 = np.random.uniform(low = -0.05, high= +0.05, size= ((l2,l3))) 
        self.weights3 = np.random.uniform(low = -0.05, high= +0.05, size= ((l3, 3)))

    def __sigmoid(self, x):
        return 1/(1+exp(-x))

    # derivative of sigmoid function
    def __sigmoid_derivative(self, x):
        return x*(1-x)

    # train neural network
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        correct = 0
        accuracy_on_train = []
        total = 0
        for epoch in range(number_of_training_iterations):     # StochasticGD implementation
            for samples in range(training_set_inputs.shape[0]):
                
                training_set_inputs_ = training_set_inputs[samples:samples+1]
                training_set_outputs_ = training_set_outputs[samples:samples+1]

                a2 = self.__sigmoid(dot(training_set_inputs_, self.weights1))
                a3 = self.__sigmoid(dot(a2, self.weights2))
                output = self.__sigmoid(dot(a3, self.weights3))

                # calculate 'error'
                del4 = (training_set_outputs_ - output)*self.__sigmoid_derivative(output)

                # find 'errors' in each layer
                del3 = dot(self.weights3, del4.T)*(self.__sigmoid_derivative(a3).T)
                del2 = dot(self.weights2, del3)*(self.__sigmoid_derivative(a2).T)

                # get gradients for each layer
                gradient3 = dot(a3.T, del4)
                gradient2 = dot(a2.T, del3.T)
                gradient1 = dot(training_set_inputs_.T, del2.T)

                # adjust weights accordingly
                self.weights1 += self.eta * gradient1
                self.weights2 += self.eta * gradient2
                self.weights3 += self.eta * gradient3
                
            for i in range(training_set_inputs.shape[0]):
        
                output = neural_network.forward_pass(training_set_inputs[i])
                prediction = np.zeros(output.shape)
                prediction[np.argmax(output)] = 1
                if (np.argmax(labels_[i]) == np.argmax(prediction)):
                    correct+=1
                else:
                    correct+=0
                total += 1

            accuracy = correct / total
            print("epoch: ", epoch, ", accuracy: ", accuracy)
            accuracy_on_train.append(accuracy)
            
        # Plots for training process:accuracy
        plt.figure(0)
        plt.plot(accuracy_on_train,'r')
        plt.xticks(np.arange(0, 8000, 100.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Num of Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend(['train'])

    def forward_pass(self, inputs):
        # pass inputs through neural network
        a2 = self.__sigmoid(dot(inputs, self.weights1))
        a3 = self.__sigmoid(dot(a2, self.weights2))
        output = self.__sigmoid(dot(a3, self.weights3)) 
        return output

if __name__ == "__main__":
    rnd_ind = random.permutation(150)
    #load the iris dataset
    iris = load_iris()
    #inputs contain 4 features
    X = iris.data[:, 0:4]
    #preprocessing of input
    X = X/np.max(X)
    #labels
    y = iris.target
   
    '''
    #One Hot Encode Y
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(y)
    '''    
    y_ = y.reshape((len(x), 1))
    enc = OneHotEncoder()
    enc.fit(y_)
    labels_ = enc.transform(y_).toarray()   # one hot encoding of y
    
    neural_network = NeuralNetwork()

    print ("Random starting weights (layer 1): ")
    print (neural_network.weights1)
    print ("\nRandom starting weights (layer 2): ")
    print (neural_network.weights2)
    print ("\nRandom starting weights (layer 3): ")
    print (neural_network.weights3)

    # the training set   
    training_set_inputs = X
    training_set_outputs = labels_

    neural_network.train(training_set_inputs, training_set_outputs, 8000)

    print ("\nNew weights (layer 1) after training: ")
    print (neural_network.weights1)
    print ("\nNew weights (layer 2) after training: ")
    print (neural_network.weights2)
    print ("\nNew weights (layer 3) after training: ")
    print (neural_network.weights3)


    '''
    temp = np.zeros(3)
    for i in range(output.shape[0]):
        temp[np.argmax(output[i,:])] = 1
        prediction[i] = temp
       
    '''
  
        
    
