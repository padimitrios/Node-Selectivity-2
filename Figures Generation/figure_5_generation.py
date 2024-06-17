import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def extract_overlaps(file_name, number_of_nodes):
    data = pd.read_csv(file_name, header=None, names=[
        'model', 'random', 'worst_overlap', 'time', 'nodes', 'filters'])

    # Convert string representations of lists to actual lists
    data['model'] = data['model'].apply(lambda x: ast.literal_eval(x))
    data['random'] = data['random'].apply(lambda x: ast.literal_eval(x))

    # Filter data based on number_of_nodes
    data_filtered = data[data['nodes'] == number_of_nodes]

    averages = {}

    for num_filters in [10, 30]:  # Exclude 50
        data_filters = data_filtered[data_filtered['filters'] == num_filters]

        best_case = np.mean([lst[0] for lst in data_filters['model']])
        our_selection_avg = np.mean([np.mean(lst) for lst in data_filters['model']])
        random_model_avg = np.mean([np.mean(lst) for lst in data_filters['random']])
        worst_case_avg = np.mean(data_filters['worst_overlap'])

        averages[num_filters] = {
            'best_case': best_case,
            'our_selection': our_selection_avg,
            'random_model': random_model_avg,
            'worst_case': worst_case_avg
        }

    return averages

# Extract overlaps for 200 nodes
overlaps_200 = extract_overlaps('results_4.csv', 200)

# Plotting for 200 nodes
plt.figure(figsize=(16, 6))  # Increase figure size horizontally
colors = ['g', 'b', 'm', 'r']
markers = ['o', 'x', '^', 's']
linestyles = ['-', '--', '-.', ':']

for i, (num_filters, overlap_dict) in enumerate(overlaps_200.items()):
    plt.plot([10, 30], [overlap_dict['best_case']] * 2, linestyle=linestyles[i], marker=markers[i],
             color=colors[0], label=f'Best Case: {num_filters} Dimensions')
    plt.plot([10, 30], [overlap_dict['our_selection']] * 2, linestyle=linestyles[i], marker=markers[i],
             color=colors[1], label=f'Proposed method: {num_filters} Dimensions')
    plt.plot([10, 30], [overlap_dict['random_model']] * 2, linestyle=linestyles[i], marker=markers[i],
             color=colors[2], label=f'Random Migration Model: {num_filters} Dimensions')
    plt.plot([10, 30], [overlap_dict['worst_case']] * 2, linestyle=linestyles[i], marker=markers[i],
             color=colors[3], label=f'Worst Case: {num_filters} Dimensions')

plt.yticks(np.arange(4, 5.6, step=0.1), fontsize=12)
plt.xticks([10, 30], fontsize=12)  # Set font size for x-axis ticks
plt.xlabel('Number of Filters', fontsize=14)  # Increase font size for x-axis label
plt.ylabel('Overlap', fontsize=14)  # Increase font size for y-axis label
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the layout
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
plt.savefig('Figures/plot_200_nodes_diff_num_filters.eps', format='eps', bbox_inches="tight")
plt.show()
plt.close()
