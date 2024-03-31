import os
import sys
import numpy as np 
import matplotlib.pyplot as plt

path = 'results/res.txt'
matrix = np.loadtxt(path)
plt.imshow(matrix, cmap="gray")
plt.savefig('results/res.png')