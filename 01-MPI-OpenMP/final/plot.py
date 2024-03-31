import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def read_submatrix(file_path):
    return np.loadtxt(file_path)

def get_submatrix_shape(file_path):
    submatrix = read_submatrix(file_path)
    return submatrix.shape

def reconstruct_matrix(rows, cols, submatrix_shape, num):
    big_matrix = np.zeros((rows * submatrix_shape[0], cols * submatrix_shape[1]))

    for i in range(rows):
        for j in range(cols):
            file_path = f"results/data/data_{i*cols + j}_{num}"
            submatrix = read_submatrix(file_path)
            big_matrix[i*submatrix_shape[0]:(i+1)*submatrix_shape[0], 
                       j*submatrix_shape[1]:(j+1)*submatrix_shape[1]] = submatrix

    return big_matrix

if len(sys.argv) != 3:
        print("Usage: python script_name.py [rows] [cols]")
        sys.exit(1)

rows = int(sys.argv[1])
cols = int(sys.argv[2])
# Assuming the first file exists and determining the shape of the submatrices
for num in ["000", "001", "002"]:
    first_file_path = f"results/data/data_0_{num}"
    if os.path.exists(first_file_path):
        submatrix_shape = get_submatrix_shape(first_file_path)
        big_matrix = reconstruct_matrix(rows, cols, submatrix_shape, num)

        # Plotting
        img_path = f"results/images/{num}.png"
        plt.imshow(big_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Reconstructed Matrix")
        plt.savefig(img_path)
        plt.close()
    else:
        print("First submatrix file not found. Please check the file path.")
