"""Varying RBF code"""

import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import numpy as np


def rbf(x, c, sigma=1.0):
    distance_squared = np.sum((x - c)**2)
    return np.exp(-distance_squared / (2 * sigma**2))


# Loading the data
dataset = torch.load('ex_dataset.pkl')
rows, cols = dataset.size()

# Preprocessing the data
filtered_dataset = []
test_dataset_negative_100 = []

for i in range(rows):
    for j in range(cols):
        z = dataset[i, j]
        x = j
        y = i
        if z == -100:
            test_dataset_negative_100.append([x, y, z])
        else:
            filtered_dataset.append([x, y, z])

non_negative_100_tensor = torch.tensor(filtered_dataset)
negative_100_tensor = torch.tensor(test_dataset_negative_100)

train_data, val_data = train_test_split(
    non_negative_100_tensor.numpy(), test_size=0.2, random_state=42)

train_tensor = torch.tensor(train_data, dtype=torch.float32)
val_tensor = torch.tensor(val_data, dtype=torch.float32)
negative_100_tensor = torch.tensor(
    test_dataset_negative_100, dtype=torch.float32)

x_train = train_tensor[:, :2]
y_train = train_tensor[:, 2]
x_val = val_tensor[:, :2]
y_val = val_tensor[:, 2]

# Fitting a linear model
solution = torch.linalg.lstsq(x_train, y_train)
A = solution.solution

predictions_val = x_val @ A
mse_val = round(((predictions_val - y_val) ** 2).mean().item(), 3)
rmse_val = round(math.sqrt(mse_val), 3)

# Testing the linear model
x_test = negative_100_tensor[:, :2]
y_test = negative_100_tensor[:, 2]
predictions_test = x_test @ A

mse_test = round(((predictions_test - y_test) ** 2).mean().item(), 3)
rmse_test = round(math.sqrt(mse_test), 3)

# Experimenting with different numbers of centers for RBF
num_centers_range = [5, 10, 15, 20, 25]
rmse_vals_rbf = []
rmse_tests_rbf = []

for num_centers in num_centers_range:
    np.random.seed(0)  # For reproducibility
    centers_indices = np.random.choice(
        x_train.shape[0], num_centers, replace=False)
    centers = x_train[centers_indices]

    phi_train = np.array([[rbf(x.numpy(), c.numpy())
                           for c in centers] for x in x_train])
    phi_val = np.array([[rbf(x.numpy(), c.numpy())
                         for c in centers] for x in x_val])
    phi_test = np.array([[rbf(x.numpy(), c.numpy()) for c in centers]
                         for x in negative_100_tensor[:, :2]])

    solution = torch.linalg.lstsq(torch.tensor(
        phi_train, dtype=torch.float32), y_train)
    A_rbf = solution.solution

    predictions_val_rbf = torch.tensor(phi_val, dtype=torch.float32) @ A_rbf
    predictions_test_rbf = torch.tensor(phi_test, dtype=torch.float32) @ A_rbf

    rmse_val_rbf = round(
        math.sqrt(((predictions_val_rbf - y_val) ** 2).mean().item()), 3)
    rmse_test_rbf = round(math.sqrt(
        ((predictions_test_rbf - negative_100_tensor[:, 2]) ** 2).mean().item()), 3)

    rmse_vals_rbf.append(rmse_val_rbf)
    rmse_tests_rbf.append(rmse_test_rbf)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(num_centers_range, rmse_vals_rbf, '-o',
         label='Validation Set RMSE (RBF)')
plt.plot(num_centers_range, rmse_tests_rbf, '-o', label='Test Set RMSE (RBF)')
plt.xlabel('Number of Centers')
plt.ylabel('RMSE')
plt.title('Effect of Number of Centers on RBF Model Performance')
plt.grid(True)
plt.legend()
plt.savefig('rbf_centers_effect.png')
plt.show()
