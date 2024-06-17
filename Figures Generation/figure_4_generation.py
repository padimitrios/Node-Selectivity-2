import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('results_2.csv', header=None, names=['model', 'random', 'worst_overlap', 'time', 'nodes'])

# Step 2: Parse the arrays from the strings
data['model'] = data['model'].apply(lambda x: list(map(float, x.strip('[]').split(', '))))
data['random'] = data['random'].apply(lambda x: list(map(float, x.strip('[]').split(', '))))

# Step 3: Filter rows for nodes
data_filtered_100 = data[data['nodes'] == 100]
data_filtered_200 = data[data['nodes'] == 200]
data_filtered_300 = data[data['nodes'] == 300]
data_filtered_400 = data[data['nodes'] == 400]
data_filtered_500 = data[data['nodes'] == 500]

# Step 4: Average time for each node batch with communication cost
def adjusted_time(avg_time, nodes):
    return (avg_time) + (nodes * 0.0002)

our_model_avg_100 = adjusted_time(np.mean(data_filtered_100['time'].tolist()), 100)
our_model_avg_200 = adjusted_time(np.mean(data_filtered_200['time'].tolist()), 200)
our_model_avg_300 = adjusted_time(np.mean(data_filtered_300['time'].tolist()), 300)
our_model_avg_400 = adjusted_time(np.mean(data_filtered_400['time'].tolist()), 400)
our_model_avg_500 = adjusted_time(np.mean(data_filtered_500['time'].tolist()), 500)

random_model = 0.3

# Step 5: Plot the results
nodes = [100, 200, 300, 400, 500]
our_model_avgs = [our_model_avg_100, our_model_avg_200, our_model_avg_300, our_model_avg_400, our_model_avg_500]
random_model_avgs = [adjusted_time(random_model, 100), adjusted_time(random_model, 200), adjusted_time(random_model, 300), adjusted_time(random_model, 400), adjusted_time(random_model, 500)]

plt.figure(figsize=(12, 6))

plt.plot(nodes, our_model_avgs, marker='o', linestyle='-', color='b', label='Proposed method')
plt.plot(nodes, random_model_avgs, marker='o', linestyle='-', color='m', label='Random Migration Model')

# Setting labels for x and y axis with larger font sizes
plt.xlabel('Number of EC nodes', fontsize=14)
plt.ylabel('Time (s)', fontsize=14)

# Adding a legend with larger font size
plt.legend(fontsize=12)

# Enabling grid
plt.grid(True)

# Setting custom x-axis ticks and larger font size
plt.xticks(nodes, fontsize=12)
plt.yticks(fontsize=12)

# Adjusting the size of the ticks
plt.tick_params(axis='both', which='major', labelsize=12)

plt.savefig('Figures/times.eps', format='eps', bbox_inches="tight")

plt.show()
