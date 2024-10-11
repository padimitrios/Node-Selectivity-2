import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the data
data = pd.read_csv('../Results/results_5.csv', header=None, names=[
                   'model', 'random', 'worst_overlap', 'time', 'nodes', 'k'])

# Step 2: Parse the arrays from the strings
data['model'] = data['model'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))
data['random'] = data['random'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))

# Define theta calculation function
def calculate_theta(values, k):
    best_option = values[0]  # First element is the best option
    theta = best_option * 0.98 - (k * 0.025)  # Adjusting to ensure higher k lowers theta
    return theta

# Initialize lists for K, theta values, and node counts
K_values = [5, 10, 15, 20, 25, 30]
theta_values = []
node_counts = []

# Calculate theta values and node counts for each K
for k in K_values:
    data_filtered = data[data['k'] == k]
    model_values = data_filtered['model'].tolist()

    # Calculate the theta value once for the current K
    theta = calculate_theta(model_values[0], k)
    theta_values.append(theta)

    # Calculate the number of nodes with overlap above theta for each row
    counts_above_theta = []
    for values in model_values:
        count_above_theta = sum(value > theta for value in values)
        counts_above_theta.append(count_above_theta)

    # Store the average count of nodes above theta for the current K
    node_counts.append(np.mean(counts_above_theta))

# Convert lists to arrays for meshgrid
K_values = np.array(K_values)
theta_values = np.array(theta_values)

# Create a mesh grid for K and theta values
K_mesh, Theta_mesh = np.meshgrid(K_values, theta_values)

# Create the Z values (node counts above theta) for the surface plot
Z = np.zeros_like(K_mesh, dtype=float)

for i in range(len(K_values)):
    for j in range(len(theta_values)):
        Z[j, i] = sum(data[data['k'] == K_values[i]]['model'].apply(lambda x: sum(v > theta_values[j] for v in x)))/len(data[data['k'] == K_values[i]])

print(Z)
print(theta_values)
print(K_values)

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(K_mesh, Theta_mesh, Z, cmap='viridis')

# Set the labels with adjusted font sizes
ax.set_xlabel('K', fontsize=12)
ax.set_ylabel('Theta', fontsize=12)
ax.set_zlabel('Nodes Selected', fontsize=12)

# Set tick parameters with adjusted label sizes
ax.tick_params(axis='both', which='major', labelsize=11)

# Rotate the plot to better visualize the relationship
ax.view_init(elev=25, azim=140)

# Adjust layout before saving
plt.tight_layout()

# Save the plot as EPS with a custom filename
plt.savefig('Figures/nodes_above_theta.eps', format='eps')

plt.show()
