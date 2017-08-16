# -*- coding: utf-8 -*-
'''
    Title:      Logistic Regression
    Author:     Minho Menezes
    
    This is the code to perform a simple Logistic Regression on a simplified Iris Dataset.
'''

# Libraries
import numpy as np 
import matplotlib.pyplot as plt

# Functions
def sigmoid(x):
    ''' Returns the Sigmoid Function applied to a vector (or value) x '''
    return 1 / (1 + np.exp(-1 * x))
    
def h_theta(x, theta):
    ''' Apply the Linear Model for features X and parameters theta '''
    return sigmoid(np.dot(np.transpose(theta), x))

def errorFunction(errors):
    ''' Calculate the Least Square Error '''
    return (1 / np.size(errors)) * np.sum(errors ** 2)

def modelAccuracy(x, y, theta):
    ''' Calculates the percentage of correct classifications '''
    return
    
def decisionBound(theta, x):
    ''' Calculates and returns a linear Decision Boundary from the model '''
    boundary_X = np.linspace(min(x[1,:]), max(x[1,:]))
    boundary_Y = -1*(theta[1] / theta[2]) * boundary_X - (theta[0] / theta[2]);
    return [boundary_X, boundary_Y]

#  Main Function
if __name__=='__main__':
   
    ###############################
    # Part 1: Data Pre-Processing #
    ###############################
    # Loads the data
    data = np.loadtxt("datasets/irisDataSimple.txt")
    
    n_examples = np.size(data,0)
    n_features = np.size(data,1) - 2
    
    # Define the model parameters
    x = np.array([np.ones(n_examples), data[:, 0], data[:, 1]])
    y = data[:, 4]
    theta = np.zeros([np.size(x, 0), 1])
    
    # Defines the hyperparameters and training measurements
    alfa = 0.1
    max_epochs = 500000
    
    error_hist = np.zeros([max_epochs])
    epsilon = 0.001
    
    ######################################
    # Part 2: Linear Regression Training #
    ######################################
    for epochs in range(max_epochs):
        # Calculate the error vector from the current Model
        y_pred = h_theta(x, theta)
        error = y_pred - y
        
        # Append new Least Square Error to History
        error_hist[epochs] = errorFunction(error)

        # Perform Gradient Descent
        for j in range(n_features):
            theta[j] = theta[j] - (alfa/n_examples) * np.sum(error * x[j,:])

        # Prints training status at each 50 epochs    
        if(epochs % 50 == 0):
            print("###### Epoch", epochs, "######")
            print("Error:", error_hist[epochs])
            print("Thetas:\n", theta)
            print("")
        
        # Evaluate convergence and stops training if so
        if(abs(error_hist[epochs] - error_hist[epochs-50]) <= epsilon):
            print("Gradient Converged!!!\nStopping at epoch", epochs)
            break
    
    #############################################
    # Part 3: Data Plotting and Training Result #
    #############################################
    # First Figure: Dataset plotting
    plt.figure(1)
    
    plt.title("Simplified Iris Dataset Classification\n(Green=Iris-setosa; Blue=Iris-virginica)")
    plt.xlabel("Sepal width (cm)")
    plt.ylabel("Sepal length (cm)")
    
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    
    plt.grid()
    plt.plot(x[1,pos], x[2,pos], 'go', x[1,neg], x[2,neg], 'bo')
    
    plt.show()

    # Second Figure: Training results
    plt.figure(2)
    deciBound = decisionBound(theta, x)
    
    plt.subplot(1,2,1)
    
    plt.title("Decision Boundary\n(Green=Iris-setosa; Blue=Iris-virginica; Black=DecisionBoundary)")
    plt.xlabel("Sepal width (cm)")
    plt.ylabel("Sepal length (cm)")
    
    plt.grid()
    plt.plot(x[1,pos], x[2,pos], 'go', x[1,neg], x[2,neg], 'bo', deciBound[0], deciBound[1], 'k-')
    
    plt.subplot(1,2,2)
    
    plt.title("Error History")
    plt.xlabel("Epochs")
    plt.ylabel("Least Square Error")
    
    plt.grid()
    plt.plot(error_hist[:epochs], "g-")
    
    plt.show()

#__