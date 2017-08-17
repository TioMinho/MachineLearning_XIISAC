# -*- coding: utf-8 -*-
'''
    Title:      Linear Regression
    Author:     Minho Menezes
    
    This is the code to perform a simple Linear Regression on a artificial generated dataset.
'''

# Libraries
import numpy as np 
import matplotlib.pyplot as plt

# Functions
def h_theta(x, theta):
    ''' Apply the Linear Model for features X and parameters theta '''
    return np.dot(np.transpose(theta), x)

def errorFunction(errors):
    ''' Calculate the Least Square Error '''
    return (1 / np.size(errors)) * np.sum(errors ** 2)   
 
# Main Function
if __name__=='__main__':
    
    ###############################
    # Part 1: Data Pre-Processing #
    ###############################
    # Loads the data
    data = np.loadtxt("datasets/cricketData.txt")
    
    n_examples = np.size(data,0)
    n_features = np.size(data,1)
    
    # Define the model parameters
    x = np.array([np.ones(n_examples), data[:, 0]])
    y = data[:, 1]
    theta = np.zeros([np.size(x, 0), 1])
    
    # Defines the hyperparameters and training measurements
    alfa = 0.005
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
        error_hist[epochs] = errorFunction(error)

        # Perform Gradient Descent
        for j in range(n_features):
            theta[j] = theta[j] - (alfa/n_examples) * np.sum(error * x[j,:])

        # Prints training status at each 100 epochs
        if(epochs % 500 == 0):
            print("###### Epoch", epochs, "######")
            print("Error:", error_hist[epochs])
            print("Thetas:\n", theta)
            print("")
        
        # Evaluate convergence and stops training if so
        if(abs(error_hist[epochs] - error_hist[epochs-50]) <= epsilon):
            print("Gradient Converged!!!\nStopping at epoch", epochs)
            print("###### Epoch", epochs, "######")
            print("Error:", error_hist[epochs])
            print("Thetas:\n", theta)
            print("")
            break
            
    #############################################
    # Part 3: Data Plotting and Training Result #
    #############################################
    # First Figure: Dataset plotting
    plt.figure(1)
    
    plt.title("Influence of Temperature on Cricket Chirp Rate")
    plt.xlabel("Rate of Cricket Chirping")
    plt.ylabel("Temperature (ºF)")
    
    plt.grid()
    plt.plot(x[1,:], y, 'rx')
    
    plt.show()
    
    # Second Figure: Training results
    plt.figure(2)
    
    plt.subplot(1,2,1)
    plt.title("Linear Regression Function Prediction\n(Black=ModelPrediction)")
    plt.xlabel("Rate of Cricket Chirping")
    plt.ylabel("Temperature (ºF)")
    
    plt.grid()
    plt.plot(x[1,:], y, 'rx', x[1,:], h_theta(x, theta)[0,:], 'k-')
    
    plt.subplot(1,2,2)
    plt.title("Error History")
    plt.xlabel("Epochs")
    plt.ylabel("Least Square Error")
    
    plt.grid()
    plt.plot(error_hist[:epochs], "g-")
    
    plt.show()
    
#__