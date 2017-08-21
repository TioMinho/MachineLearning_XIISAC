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

def accuracyFunction(X, y, theta):
    ''' Calculates the percentage of correct classifications '''
    Y_pred = h_theta(X, theta)
    pos = np.where(Y_pred >= 0.5); Y_pred[pos] = 1
    neg = np.where(Y_pred < 0.5); Y_pred[neg] = 0
    
    return 100 * (1 - (1. / np.size(y)) * np.sum((Y_pred - y) ** 2))
    
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
    data = np.loadtxt("../datasets/dataRegLog1.txt")
    
    n_examples = np.size(data,0)
    n_features = np.size(data,1)
    
    # Define the model parameters
    x = np.array([np.ones(n_examples), data[:, 0], data[:, 1]])
    y = data[:, -1]
    theta = np.zeros([np.size(x, 0), 1])

    # Defines the hyperparameters and training measurements
    alfa = 0.05
    max_epochs = 500000
    
    error_hist = np.zeros([max_epochs])
    epsilon = 0.01
    
    ######################################
    # Part 2: Linear Regression Training #
    ######################################
    for epochs in range(max_epochs):
        # Calculate the error vector from the current Model
        y_pred = h_theta(x, theta)
        error = y_pred - y
        
        # Append new Least Square Error to History
        error_hist[epochs] = accuracyFunction(x, y, theta)

        # Perform Gradient Descent
        for j in range(n_features):
            theta[j] = theta[j] - (alfa/n_examples) * np.sum(error * x[j,:])

        # Prints training status at each 50 epochs    
        if(epochs % 10 == 0):
            print("###### Epoch", epochs, "######")
            print("Error:", error_hist[epochs], "%")
            print("Thetas:\n", theta)
            print("")
        
        # Evaluate convergence and stops training if so
        if(abs(error_hist[epochs] - error_hist[epochs-50]) <= epsilon):
            print("Gradient Converged!!!\nStopping at epoch", epochs)
            print("###### Epoch", epochs, "######")
            print("Error:", error_hist[epochs], "%")
            print("Thetas:\n", theta)
            break
    
    #############################################
    # Part 3: Data Plotting and Training Result #
    #############################################

    x = np.linspace(-30,30)

    plt.figure(1)
    plt.title("Sigmoid Function")
    plt.xlabel("$x$")
    plt.ylabel(r"$\frac{1}{1+e^{-x}}$")

    plt.plot(x, sigmoid(x), 'b-')

    plt.show()

    # First Figure: Dataset plotting
    # plt.figure(1)
    
    # plt.title("Artificial Data for Uniclass Problem\n(Green=Class 1; Blue=Not Class 2)")
    # plt.xlabel("$X_{1}$")
    # plt.ylabel("$X_{2}$")
    
    # pos = np.where(y == 1)
    # neg = np.where(y == 0)
    
    # plt.grid()
    # plt.plot(x[1,pos], x[2,pos], 'go', x[1,neg], x[2,neg], 'bo')
    
    # plt.show()

    # # Second Figure: Training results
    # plt.figure(2)
    # deciBound = decisionBound(theta, x)
    
    # plt.title("Artificial Data for Uniclass Problem\n(Green=Class 1; Blue=Not Class 1; Black=DecisionBoundary)")
    # plt.xlabel("$X_{1}$")
    # plt.ylabel("$X_{2}$")
    # plt.ylim(-10, 5)

    # plt.grid()
    # plt.plot(x[1,pos], x[2,pos], 'go', x[1,neg], x[2,neg], 'bo', deciBound[0], deciBound[1], 'k-')
    
    # plt.show()

#__