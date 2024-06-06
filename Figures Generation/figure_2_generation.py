import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('200N_5K.csv', header=None, names=[
                   'model', 'random', 'worst_overlap', 'time', 'dimension'])

# Step 2: Parse the arrays from the strings
data['model'] = data['model'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))
data['random'] = data['random'].apply(
    lambda x: list(map(float, x.strip('[]').split(', '))))

# Step 3: Filter rows for dimensions 5, 10, and 20
data_filtered_5 = data[data['dimension'] == 5]
data_filtered_10 = data[data['dimension'] == 10]
data_filtered_20 = data[data['dimension'] == 20]

# Step 4: Calculate averages for each dimension
our_model_avg_5 = np.mean(data_filtered_5['model'].tolist())
our_model_avg_10 = np.mean(data_filtered_10['model'].tolist())
our_model_avg_20 = np.mean(data_filtered_20['model'].tolist())

random_model_avg_5 = np.mean(data_filtered_5['random'].tolist())
random_model_avg_10 = np.mean(data_filtered_10['random'].tolist())
random_model_avg_20 = np.mean(data_filtered_20['random'].tolist())

worst_overlap_5 = np.mean(data_filtered_5['worst_overlap'].tolist())
worst_overlap_10 = np.mean(data_filtered_10['worst_overlap'].tolist())
worst_overlap_20 = np.mean(data_filtered_20['worst_overlap'].tolist())

# Calculate average best case for each dimension
best_case_avg_5 = np.mean([row[0] for row in data_filtered_5['model']])
best_case_avg_10 = np.mean([row[0] for row in data_filtered_10['model']])
best_case_avg_20 = np.mean([row[0] for row in data_filtered_20['model']])

# Step 5: Plot the results
dimensions = [5, 10, 20]
our_model_avgs = [our_model_avg_5, our_model_avg_10, our_model_avg_20]
random_model_avgs = [random_model_avg_5,
                     random_model_avg_10, random_model_avg_20]
worst_overlaps = [worst_overlap_5, worst_overlap_10, worst_overlap_20]
best_case = [best_case_avg_5, best_case_avg_10, best_case_avg_20]

plt.figure(figsize=(10, 5))

plt.plot(dimensions, best_case, marker='x', linestyle='-',
         color='g', label='Best Case Average')
plt.plot(dimensions, our_model_avgs, marker='o', linestyle='-',
         color='b', label='Our Model Average')
plt.plot(dimensions, random_model_avgs, marker='x', linestyle='-',
         color='m', label='Random Model Average')
plt.plot(dimensions, worst_overlaps, marker='o', linestyle='-',
         color='r', label='Worst Case Average')


# Setting labels for x and y axis
plt.xlabel('Dimensions')
plt.ylabel('Overlap')

# Adding legend
plt.legend()

# Enabling grid
plt.grid(True)

# Setting custom x-axis ticks
plt.xticks(dimensions)


# Save the figure as .eps
plt.savefig('Figures/overlap_analysis_dimensions_200N_5K.eps',
            format='eps', bbox_inches="tight")

# Display the plot
plt.show()
