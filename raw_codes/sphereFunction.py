# -*- coding: utf-8 -*-
# Libraries
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def sphereFunction(x):
    return np.sum(x ** 2)

# Main Function
if __name__=='__main__':
    
    x = np.linspace(-5.12, 5.12)
    y = np.linspace(-5.12, 5.12)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    plt.title("Error Function")
    ax.set_xlabel(r"$ \theta_{0} $")
    ax.set_ylabel(r"$ \theta_{1} $")
    ax.set_zlabel(r"$ J(\theta) $")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    X, Y = np.meshgrid(x, y)

    ax.plot_surface(Y,X,X**2+Y**2, cmap=cm.Spectral)

    plt.show()

#__