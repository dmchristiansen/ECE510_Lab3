#Import necessary packages
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential #Sequential Models
from keras.layers import Dense #Dense Fully Connected Layer Type
from keras.optimizers import SGD #Stochastic Gradient Descent Optimizer

#load the iris dataset
iris = load_iris()
#inputs contain 4 features
X = iris.data[:, 0:4]
X = X/np.max(X)
#labels
y = iris.target
#print the distinct y labels
print(np.unique(y))

#One Hot Encode Y
encoder = LabelBinarizer()
Y = encoder.fit_transform(y)

def baseline_network():
    model = Sequential()
    model.add(Dense(3, input_shape=(4,), activation='sigmoid'))
    model.add(Dense(3, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
        
    #stochastic gradient descent
    sgd = SGD(lr=0.1, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

neural_network = baseline_network()
history = neural_network.fit(X,Y, epochs=1000)

np.set_printoptions(suppress=True)
prediction = neural_network.predict(X, batch_size=1, verbose=0)

plt.figure(0)
plt.plot(history.history['acc'],'r')
plt.xticks(np.arange(0, 500, 100.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.legend(['train'])
