import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('results_2.csv', header=None, names=[
                   'model', 'random', 'worst_overlap', 'time', 'nodes'])

# Step 2: Parse the arrays from the strings
data['model'] = data['model'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))
data['random'] = data['random'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))

# Step 3: Filter rows for nodes
data_filtered_100 = data[data['nodes'] == 100]
data_filtered_200 = data[data['nodes'] == 200]
data_filtered_300 = data[data['nodes'] == 300]
data_filtered_400 = data[data['nodes'] == 400]
data_filtered_500 = data[data['nodes'] == 500]

# average time for each node batch
our_model_avg_100 = np.mean(data_filtered_100['time'].tolist())
our_model_avg_200 = np.mean(data_filtered_200['time'].tolist())
our_model_avg_300 = np.mean(data_filtered_300['time'].tolist())
our_model_avg_400 = np.mean(data_filtered_400['time'].tolist())
our_model_avg_500 = np.mean(data_filtered_500['time'].tolist())

random_model = 0.0069570940

# Step 5: Plot the results
nodes = [100, 200, 300, 400, 500]
our_model_avgs = [our_model_avg_100, our_model_avg_200,
                  our_model_avg_300, our_model_avg_400, our_model_avg_500]
random_model_avgs = [random_model, random_model,
                     random_model, random_model, random_model]

plt.figure(figsize=(10, 5))

plt.plot(nodes, our_model_avgs, marker='o', linestyle='-',
         color='b', label='Our Model Average')
plt.plot(nodes, random_model_avgs, marker='x', linestyle='-',
         color='m', label='Random Model Average')

# Setting labels for x and y axis
plt.xlabel('Nodes')
plt.ylabel('Time (s)')

# Adding a legend
plt.legend()

# Enabling grid
plt.grid(True)

# Setting custom x-axis ticks
plt.xticks(nodes)

plt.savefig('Figures/times.eps',
            format='eps', bbox_inches="tight")

plt.show()
