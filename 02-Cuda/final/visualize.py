import os
import sys
import numpy as np 
import matplotlib.pyplot as plt

paths = os.listdir("results")
for path in paths:
    p = f'results/{path}/res.txt'
    matrix = np.loadtxt(p)
    plt.imshow(matrix, cmap="gray")
    plt.savefig(f'results/{path}/res.png')