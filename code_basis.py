"""Main code with basis function"""

import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import numpy as np


def rbf(x, c, sigma=1.0):
    distance_squared = np.sum((x - c)**2)
    return np.exp(-distance_squared / (2 * sigma**2))


"""Loading the data as per question using torch module"""
dataset = torch.load('ex_dataset.pkl')

rows, cols = dataset.size()

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


# Fitting a linear model using linear least squares approach
solution = torch.linalg.lstsq(x_train, y_train)
A = solution.solution
residuals = solution.residuals

predictions_val = x_val @ A

# Calculating metrics for validation set
mse_val = round(((predictions_val - y_val) ** 2).mean().item(), 3)
rmse_val = round(math.sqrt(mse_val), 3)

print(f"Validation Set Metrics:")
print(f"Mean Squared Error (MSE): {mse_val}")
print(f"Root Mean Squared Error (RMSE): {rmse_val}")

# Testing the model using negative_100_tensor enxposed
x_test = negative_100_tensor[:, :2]
y_test = negative_100_tensor[:, 2]
predictions_test = x_test @ A

# Calculating metrics for test set
mse_test = round(((predictions_test - y_test) ** 2).mean().item(), 3)
rmse_test = round(math.sqrt(mse_test), 3)

print(f"\nTest Set Metrics:")
print(f"Mean Squared Error (MSE): {mse_test}")
print(f"Root Mean Squared Error (RMSE): {rmse_test}")


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotting the entire dataset
ax.scatter(torch.cat([x_train[:, 0], x_val[:, 0], negative_100_tensor[:, 0]]).numpy(),
           torch.cat([x_train[:, 1], x_val[:, 1],
                     negative_100_tensor[:, 1]]).numpy(),
           torch.cat([y_train, y_val, negative_100_tensor[:, 2]]).numpy(),
           color='grey', label='Data Points')


mask_val = y_val == -100
mask_test = y_test == -100

ax.scatter(predictions_val[mask_val].numpy(),
           x_val[:, 0][mask_val].numpy(),
           x_val[:, 1][mask_val].numpy(),
           color='blue', label='Predicted Values (Validation Set)')

ax.scatter(predictions_test[mask_test].numpy(),
           negative_100_tensor[:, 0][mask_test].numpy(),
           negative_100_tensor[:, 1][mask_test].numpy(),
           color='red', label='Predicted -100 Values (Test Set)')

ax.set_xlabel('x values')
ax.set_ylabel('y values')
ax.set_zlabel('z values')
ax.set_title(
    '3D Plot of Linear Model Predictions with Highlighted Predicted -100 Values')
ax.legend()
plt.savefig('3d_plot.png')
plt.show()


# Select 10 centers randomly from the training data
np.random.seed(0)  # For reproducibility
centers = x_train[np.random.choice(x_train.shape[0], 10, replace=False)]

# Transform the 2D input data into a 10D space using RBFs
phi_train = np.array([[rbf(x.numpy(), c.numpy())
                     for c in centers] for x in x_train])
phi_val = np.array([[rbf(x.numpy(), c.numpy())
                   for c in centers] for x in x_val])
phi_test = np.array([[rbf(x.numpy(), c.numpy()) for c in centers]
                    for x in negative_100_tensor[:, :2]])

# Fitting a linear model to the new transformed data
solution = torch.linalg.lstsq(torch.tensor(
    phi_train, dtype=torch.float32), y_train)
A_rbf = solution.solution

# Predicting z values for validation and test sets
predictions_val_rbf = torch.tensor(phi_val, dtype=torch.float32) @ A_rbf
predictions_test_rbf = torch.tensor(phi_test, dtype=torch.float32) @ A_rbf

# Calculating metrics for the RBF model
mse_val_rbf = round(((predictions_val_rbf - y_val) ** 2).mean().item(), 3)
rmse_val_rbf = round(math.sqrt(mse_val_rbf), 3)
mse_test_rbf = round((
    (predictions_test_rbf - negative_100_tensor[:, 2]) ** 2).mean().item(), 3)
rmse_test_rbf = round(math.sqrt(mse_test_rbf), 3)

print("Metrics for RBF Model:")
print(f"Validation Set - Mean Squared Error (MSE): {mse_val_rbf}")
print(f"Validation Set - Root Mean Squared Error (RMSE): {rmse_val_rbf}")
print(f"Test Set - Mean Squared Error (MSE): {mse_test_rbf}")
print(f"Test Set - Root Mean Squared Error (RMSE): {rmse_test_rbf}")

# Comparing the performance of the RBF model with the purely linear model
print("\nComparison:")
print(f"Linear Model (RMSE): {rmse_test}")
print(f"RBF Model (RMSE): {rmse_test_rbf}")

# Plotting the predictions of the RBF model on the test set
# Plotting the predictions of the RBF model on the test set in 3D
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(negative_100_tensor[:, 0].numpy(), negative_100_tensor[:, 1].numpy(
# ), negative_100_tensor[:, 2].numpy(), color='red', label='Actual Test Data')
# ax.scatter(negative_100_tensor[:, 0].numpy(), negative_100_tensor[:, 1].numpy(
# ), predictions_test_rbf.numpy(), color='blue', label='Predicted Test Data (RBF Model)')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('RBF Model Predictions vs Actual (Test Set)')
# ax.legend()
# plt.savefig('3d_plot_compariosn.png')

# plt.show()


# Plotting the entire dataset with the predicted values (corresponding to the -100s) highlighted in 3D for RBF model
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotting the entire dataset
ax.scatter(torch.cat([x_train[:, 0], x_val[:, 0], negative_100_tensor[:, 0]]).numpy(),
           torch.cat([x_train[:, 1], x_val[:, 1],
                     negative_100_tensor[:, 1]]).numpy(),
           torch.cat([y_train, y_val, negative_100_tensor[:, 2]]).numpy(),
           color='grey', label='Data Points')

# Highlighting the predicted values corresponding to -100s from RBF model
mask_rbf = negative_100_tensor[:, 2] == -100
ax.scatter(negative_100_tensor[:, 0][mask_rbf].numpy(),
           negative_100_tensor[:, 1][mask_rbf].numpy(),
           predictions_test_rbf[mask_rbf].numpy(),
           color='blue', label='Predicted -100 Values (RBF Model)')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of Data Points with Predictions (RBF Model)')
ax.legend()
plt.savefig('rbf3d')

plt.show()
