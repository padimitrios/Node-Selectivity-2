import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load the data
data = pd.read_csv('results_5.csv', header=None, names=['model', 'random', 'worst_overlap', 'time', 'nodes', 'k'])

# Step 2: Parse the arrays from the strings
data['model'] = data['model'].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
data['random'] = data['random'].apply(lambda x: list(map(float, x.strip('[]').split(', '))))

# Initialize lists to store pass and fail counts for each K
pass_counts = []
fail_counts = []

# Define theta calculation function
def calculate_theta(values):
    best_option = values[0]  # First element is the best option
    theta = best_option * 0.92  # 20% less than the best option
    return theta

# Loop through each K value
for k in [5, 10, 15, 20, 25, 30]:
    # Filter data for current K value
    data_filtered = data[data['k'] == k]
    
    # Calculate theta for current K value
    thetas = data_filtered['model'].apply(calculate_theta)
    
    # Initialize counters for pass and fail
    pass_count = 0
    fail_count = 0
    
    # Iterate through each row in filtered data
    for idx, row in data_filtered.iterrows():
        model_values = row['model']
        theta = thetas[idx]  # Get theta for current row
        
        # Count how many model values pass and fail the threshold
        for value in model_values:
            if value >= theta:
                pass_count += 1
            else:
                fail_count += 1
    
    pass_counts.append(pass_count)
    fail_counts.append(fail_count)

# Extract K values for x-axis
K_values = [5, 10, 15, 20, 25, 30]

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(K_values, pass_counts, color='blue', label='Pass')
plt.bar(K_values, fail_counts, bottom=pass_counts, color='red', label='Fail')
plt.xlabel('K Values')
plt.ylabel('Number of Results')
plt.title('Results Passing/Failing Threshold vs K Values')
plt.xticks(K_values)
plt.legend()
plt.grid(True)
plt.show()
