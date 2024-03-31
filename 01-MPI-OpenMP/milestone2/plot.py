import csv
import os
import sys

import numpy as np 
import matplotlib.pyplot as plt
import imageio.v3 as iio

file_dir = 'results/data'
img_dir = 'results/images'
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

files = os.listdir(file_dir)
files.sort()

for file in files:
    path = os.path.join(file_dir, str(file))
    
    img = os.path.join(img_dir, str(file) + '.png')

    matrix = np.loadtxt(path)

    # Plotting the matrix
    plt.imshow(matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar()

    plt.savefig(img)
    plt.close()
