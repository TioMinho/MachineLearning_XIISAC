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
 
def errGraph(x):
    return x**2-x

# Main Function
if __name__=='__main__':
    
    x = np.linspace(-100, 100)
    
    # First Figure: Dataset plotting
    plt.figure(1)
    
    plt.title("Error Function")
    plt.xlabel(r"$ \theta_{0} $")
    plt.ylabel(r"$ J(\theta) $")
    
    plt.xticks([])
    plt.yticks([])
    plt.grid()
    
    plt.plot(x, errGraph(x), 'g-')
    
    plt.show()
    
#__