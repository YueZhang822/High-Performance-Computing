import matplotlib.pyplot as plt
import os

log_dir_path = 'losses'
images_dir = 'images'

log_files = os.listdir(log_dir_path)

for log_file in log_files:
    epochs = []
    losses = []

    full_path = os.path.join(log_dir_path, log_file)

    with open(full_path, 'r') as file:
        for line in file:
            parts = line.split(',')
            epoch = int(parts[0].split(': ')[1])
            loss = float(parts[1].split(': ')[1])
            epochs.append(epoch)
            losses.append(loss)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.title(f'Training Loss ({log_file[:-4]})', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.grid(True)
    plt.xticks(epochs)

    plot_file_name = f"{log_file.split('.')[0]}_plot.png"
    plt.savefig(os.path.join(images_dir, plot_file_name))

    plt.clf()
