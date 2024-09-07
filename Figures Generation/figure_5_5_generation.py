import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast


def extract_overlaps(file_name, number_of_nodes):
    data = pd.read_csv(file_name, header=None, names=[
        'model', 'random', 'greedy_selection', 'worst_overlap', 'time', 'nodes', 'filters', 'DIM', 'K'])

    # Convert string representations of lists to actual lists
    data['model'] = data['model'].apply(lambda x: ast.literal_eval(x))
    data['random'] = data['random'].apply(lambda x: ast.literal_eval(x))
    data['greedy_selection'] = data['greedy_selection'].apply(lambda x: ast.literal_eval(x))

    # Filter data based on number_of_nodes
    data_filtered = data[data['nodes'] == number_of_nodes]

    averages = {}

    for num_filters in [10, 30, 50]:
        data_filters = data_filtered[data_filtered['filters'] == num_filters]

        best_case = np.mean([lst[0] for lst in data_filters['model']])
        our_selection_avg = np.mean([np.mean(lst[:5])
                                    for lst in data_filters['model']])
        random_model_avg = np.mean([np.mean(lst)
                                   for lst in data_filters['random']])
        greedy_selection_avg = np.mean([np.mean(lst)
                                        for lst in data_filters ['greedy_selection']])
        worst_case_avg = np.mean(data_filters['worst_overlap'])

        averages[num_filters] = {
            'best_case': best_case,
            'our_selection': our_selection_avg,
            'random_model': random_model_avg,
            'greedy_selection': greedy_selection_avg,
            'worst_case': worst_case_avg
        }

    return averages


# Extract overlaps for 200 nodes
overlaps_200 = extract_overlaps('200_greedy_k40_diff_filt.csv', 200)

# Plotting for 200 nodes
plt.figure(figsize=(16, 6))  # Increase figure size horizontally
filters = [10, 30, 50]

best_overlap = [overlaps_200[f]['best_case'] for f in filters]
our_method = [overlaps_200[f]['our_selection'] for f in filters]
random_model = [overlaps_200[f]['random_model'] for f in filters]
greedy_selection = [overlaps_200[f]['greedy_selection'] for f in filters]
worst_overlap = [overlaps_200[f]['worst_case'] for f in filters]

plt.plot(filters, best_overlap, linestyle='-',
         marker='o', color='g', label='Best Case')
plt.plot(filters, our_method, linestyle='-',
         marker='o', color='b', label='Proposed Method')
plt.plot(filters, greedy_selection, linestyle='-',
            marker='o', color='c', label='Greedy Selection')
plt.plot(filters, random_model, linestyle='-', marker='o',
         color='m', label='Random Migration Model')
plt.plot(filters, worst_overlap, linestyle='-',
         marker='o', color='r', label='Worst Case')


plt.xticks(filters, fontsize=12)  # Set font size for x-axis ticks
# Increase font size for x-axis label
plt.xlabel('Number of Filters', fontsize=14)
plt.ylabel('Overlap', fontsize=14)  # Increase font size for y-axis label
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the layout

# Position legend inside the graph, at the bottom
plt.legend()

plt.savefig('Figures/Greedy/plot_200_nodes_diff_num_filters_k40.eps',
            format='eps', bbox_inches="tight")
plt.show()
plt.close()
