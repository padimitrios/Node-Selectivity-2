import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the data
data = pd.read_csv('results_5.csv', header=None, names=[
                   'model', 'random', 'worst_overlap', 'time', 'nodes', 'k'])

# Step 2: Parse the arrays from the strings
data['model'] = data['model'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))
data['random'] = data['random'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))

# Define theta calculation function


def calculate_theta(values):
    best_option = values[0]  # First element is the best option
    theta = best_option * 0.96  # 2% less than the best option
    return theta


# Initialize lists for K and theta values
K_values = [5, 10, 15, 20, 25, 30]
theta_values = []
node_counts = []

# Calculate theta values and node counts for each K
for k in K_values:
    data_filtered = data[data['k'] == k]
    model_values = data_filtered['model'].tolist()

    # Calculate the theta value once for the current K
    theta = calculate_theta(model_values[0])
    theta_values.append(theta)

    # Calculate the number of nodes with overlap above theta for each row
    counts_above_theta = []
    for values in model_values:
        count_above_theta = sum(value > theta for value in values)
        counts_above_theta.append(count_above_theta)

    # Store the average count of nodes above theta for the current K
    node_counts.append(np.mean(counts_above_theta))

# Create a mesh grid for K and theta values
K_mesh, Theta_mesh = np.meshgrid(K_values, theta_values)
print(theta_values)

# Create the Z values (node counts above theta) for the surface plot
Z = np.zeros_like(K_mesh, dtype=float)

for i in range(len(K_values)):
    Z[:, i] = node_counts[i]

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(K_mesh, Theta_mesh, Z, cmap='viridis')

# Set the labels with adjusted font sizes
ax.set_xlabel('K Values', fontsize=14)
ax.set_ylabel('Theta Values', fontsize=14)
ax.set_zlabel('Nodes Selected', fontsize=14)

# Set tick parameters with adjusted label sizes
ax.tick_params(axis='both', which='major', labelsize=12)

# Rotate the plot so that K value of 5 is on the outside
ax.view_init(elev=20, azim=120)

# Save the plot as EPS with a custom filename
plt.savefig('Figures/nodes_above_theta.eps', format='eps', bbox_inches="tight")

plt.show()
