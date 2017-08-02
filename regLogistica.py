# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt

def errorFunction(errors):
    return (1 / np.size(errors)) * np.sum(errors ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))
    
def decisionBound(theta, x):
    boundary_X = np.linspace(min(x[1,:]), max(x[1,:]))
    boundary_Y = -1*(theta[1] / theta[2]) * boundary_X - (theta[0] / theta[2]);
    return [boundary_X, boundary_Y]
 
def h_theta(x, theta):
    return sigmoid(np.dot(np.transpose(theta), x))
 
if __name__=='__main__':
    data = np.loadtxt("irisDataSimple.txt")
    
    n_examples = np.size(data,0)
    n_features = np.size(data,1) - 2
    
    x = np.array([np.ones(n_examples), data[:, 0], data[:, 1]])
    y = data[:, 4]
    theta = np.zeros([np.size(x, 0), 1])
    
    alfa = 0.1
    max_epochs = 500000
    
    error_hist = np.zeros([max_epochs])
    epsilon = 0.001
    
    for epochs in range(max_epochs):
        y_pred = h_theta(x, theta)
        error = y_pred - y

        error_hist[epochs] = errorFunction(error)

        for j in range(n_features):
            theta[j] = theta[j] - (alfa/n_examples) * np.sum(error * x[j,:])

        print("###### Epoch", epochs, "######")
        print("Error:", error_hist[epochs])
        print("Thetas:\n", theta)
        print("")
        
        if(abs(error_hist[epochs] - error_hist[epochs-50]) <= epsilon):
            print("Gradient Converged!!!\nStopping at epoch", epochs)
            break
            
    plt.figure(1)
    plt.title("Classificação do Iris Dataset Simplificado\n(Verde=Iris-setosa; Azul=Iris-virginica)")
    plt.xlabel("Comprimento da Sépala (cm)")
    plt.ylabel("Largura da Sépala (cm)")
    
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    
    plt.grid()
    plt.plot(x[1,pos], x[2,pos], 'go', x[1,neg], x[2,neg], 'bo')
    plt.show()

    plt.figure(2)
    deciBound = decisionBound(theta, x)
    
    plt.subplot(1,2,1)
    plt.title("Classificação do Iris Dataset Simplificado\n(Verde=Iris-setosa; Azul=Iris-virginica)")
    plt.xlabel("Comprimento da Sépala (cm)")
    plt.ylabel("Largura da Sépala (cm)")
    plt.grid()
    plt.plot(x[1,pos], x[2,pos], 'go', x[1,neg], x[2,neg], 'bo', deciBound[0], deciBound[1], 'k-')
    
    plt.subplot(1,2,2)
    plt.grid()
    plt.plot(error_hist[:epochs], "g-")
    
    plt.show()
