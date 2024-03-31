"""File for initial plot of data"""

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Load the data
dataset = torch.load('ex_dataset.pkl')

rows, cols = dataset.size()


x_coords = []
y_coords = []
z_coords = []

for i in range(rows):
    for j in range(cols):
        z = dataset[i, j]
        x = j
        y = i

        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

# Convert lists to numpy arrays
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
z_coords = np.array(z_coords)

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(x_coords, y_coords, z_coords, c=z_coords,
           cmap='viridis', label='Data Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Initial Dataset')
ax.legend()
plt.savefig('initial_dataset.png')

plt.show()
