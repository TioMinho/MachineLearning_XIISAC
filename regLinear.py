# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt

def h_theta(x, theta):
	return np.dot(np.transpose(theta), x)

def errorFunction(errors):
	return (1 / np.size(errors)) * np.sum(errors ** 2)

if __name__=='__main__':
    data = np.loadtxt("cricketData.txt")
    
    n_examples = np.size(data,0)
    n_features = np.size(data,1)
    
    x = np.array([np.ones(n_examples), data[:, 0]])
    y = data[:, 1]
    theta = np.zeros([np.size(x, 0), 1])
    
    alfa = 0.005
    max_epochs = 500000
    
    error_hist = np.zeros([max_epochs])
    epsilon = 0.01
    
    for epochs in range(max_epochs):
        y_pred = h_theta(x, theta)
        error = y_pred - y

        error_hist[epochs] = errorFunction(error)

        for j in range(n_features):
            theta[j] = theta[j] - (alfa/n_examples) * np.sum(error * x[j,:])

        if(epochs % 50 == 0):
            print("###### Epoch", epochs, "######")
            print("Error:", error_hist[epochs])
            print("Thetas:\n", theta)
            print("")
        
        if(abs(error_hist[epochs] - error_hist[epochs-50]) <= epsilon):
            print("Gradient Converged!!!\nStopping at epoch", epochs)
            break
            
    plt.figure(1)
    plt.title("Influence of Temperature on Cricket Chirp Rate")
    plt.xlabel("Rate of Cricket Chirping")
    plt.ylabel("Temperature (ºF)")
    
    plt.grid()
    plt.plot(x[1,:], y, 'rx')
    plt.show()
    
    plt.figure(2)
    
    plt.subplot(1,2,1)
    plt.grid()
    plt.plot(x[1,:], y, 'rx', x[1,:], h_theta(x, theta)[0,:], 'b-')
    
    plt.subplot(1,2,2)
    plt.grid()
    plt.plot(error_hist[:epochs], "g-")
    
    plt.show()
