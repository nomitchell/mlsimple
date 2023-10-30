import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def initialize_parameters(lenw):
    w = np.random.randn(1, lenw)
    b = 0
    return w, b

def forward_prop(x, w, b):
    z = np.dot(w,x) + b
    return z

def cost_function(z, y):
    m = y.shape[1]
    J = (1/(2*m))*np.sum(np.square(z-y))
    return J

def back_prop(x, y, z):
    m = y.shape[1]
    dz = (1/m)*(z-y) 
    dw = np.dot(dz,x.T)
    db = np.sum(dz)
    return dw, db

def gradient_descent_update(w, b, dw, db, learning_rate):
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w, b

def linear_regression_model(x_train, y_train, x_val, y_val, learning_rate, epochs):
    lenw = x_train.shape[0]
    w, b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(epochs):
        z_train = forward_prop(x_train, w, b)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(x_train, y_train, z_train)
        w, b = gradient_descent_update(w, b, dw, db, learning_rate)

        if i%10==0:
            costs_train.append(cost_train)
        
        MAE_train = (1/m_train)*np.sum(np.abs(z_train-y_train))

        z_val = forward_prop(x_val, w, b)

        cost_val = cost_function(z_val, y_val)

        MAE_val = (1/m_val)*np.sum(np.abs(z_val-y_val))

        print('Epochs ' + str(i) + '/' + str(epochs) + ': ')
        print('Training Cost ' + str(cost_train) + '|' + 'Validation cost' + str(cost_val))
        print('Training MAE ' + str(MAE_train) + '|' + 'Validation cost' + str(MAE_val))

    plt.plot(costs_train)
    plt.xlabel('iterations per ten')
    plt.ylabel('cost')
    plt.title('Learning rate ' + str(learning_rate))
    plt.show()

def normalize(df):
    return (df-df.mean())/df.std()

#################################################

class linearRegression():
    '''
    Given a pandas dataframe of numerical columns, will auto-normalize and perform gradient descent,
    where predictions can then be made. Automatically presents graphs for simple data visualization.
    '''
    
    def initialize_parameters(lenw):
        w = np.random.randn(1, lenw)
        b = 0
        return w, b

    def predY(self, x, w, b):
        z = np.dot(w,x) + b
        return z

    def cost_function(z, y):
        m = y.shape[1]
        J = (1/(2*m))*np.sum(np.square(z-y))
        return J
    
    def gradientDescent(self, x, y, z, w, b, learning_rate):
        m = y.shape[1]
        dz = (1/m)*(z-y) 
        dw = np.dot(dz,x.T)
        db = np.sum(dz)

        w = w - learning_rate*dw
        b = b - learning_rate*db

        return w, b

    def cost_function(z, y):
        m = y.shape[1]
        J = (1/(2*m))*np.sum(np.square(z-y))
        return J
    
    def plotCost(self, costs_train):
        plt.plot(costs_train)
        plt.xlabel('iterations per ten')
        plt.ylabel('cost')
        plt.title('Cost plot')
        plt.show()

    def fit(self, x_train, y_train, x_val, y_val, learning_rate=0.01, epochs=1000):
        lenw = x_train.shape[0]
        w, b = initialize_parameters(lenw)

        costs_train = []

        for i in range(epochs):
            z_train = self.predY(x_train, w, b)
            cost_train = cost_function(z_train, y_train)
            w, b = self.gradientDescent(x_train, y_train, z_train, w, b, learning_rate)

            if i%10==0:
                costs_train.append(cost_train)

            z_val = self.predY(x_val, w, b)

            cost_val = cost_function(z_val, y_val)

            print('Epochs ' + str(i) + '/' + str(epochs) + ': ')
            print('Training Cost ' + str(cost_train) + '|' + 'Validation cost' + str(cost_val))

        self.plotCost(costs_train)

