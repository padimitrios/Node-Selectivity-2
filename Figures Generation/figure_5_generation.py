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

    for num_filters in [10, 30, 50]:

        data_filters = data_filtered[data_filtered['filters'] == num_filters]

        best_case = np.mean([lst[0] for lst in data_filters['model']])
        our_selection_avg = np.mean([np.mean(lst)
                                    for lst in data_filters['model']])
        random_model_avg = np.mean([np.mean(lst)
                                   for lst in data_filters['random']])
        worst_case_avg = np.mean(data_filters['worst_overlap'])

        averages[num_filters] = {
            'best_case': best_case,
            'our_selection': our_selection_avg,
            'random_model': random_model_avg,
            'worst_case': worst_case_avg
        }

    return averages


# Extract overlaps for 100 nodes
overlaps_100 = extract_overlaps('results_4.csv', 100)

# Extract overlaps for 200 nodes
overlaps_200 = extract_overlaps('results_4.csv', 200)

# Plotting for 100 nodes
plt.figure(figsize=(10, 5))
colors = ['b', 'g', 'r']
markers = ['-o', '-x', '-^', '-s']
for i, (num_filters, overlap_dict) in enumerate(overlaps_100.items()):
    plt.plot([10, 30, 50], [overlap_dict['best_case']] * 3, markers[i],
             color=colors[i], label=f'Best case: {num_filters}')
    plt.plot([10, 30, 50], [overlap_dict['our_selection']] * 3, markers[i],
             color=colors[i], linestyle='--', label=f'Our Selection: {num_filters}')
    plt.plot([10, 30, 50], [overlap_dict['random_model']] * 3, markers[i],
             color=colors[i], linestyle='-.', label=f'Random Model: {num_filters}')
    plt.plot([10, 30, 50], [overlap_dict['worst_case']] * 3, markers[i],
             color=colors[i], linestyle=':', label=f'Worst Case: {num_filters}')


plt.yticks(np.arange(4, 5.6, step=0.1))
plt.xticks([10, 30, 50])
plt.xlabel('Number of Filters')
plt.ylabel('Average Overlap')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the layout
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.savefig('Figures/plot_100_nodes_diff_num_filters.eps',
            format='eps', bbox_inches="tight")
plt.close()


# Plotting for 200 nodes
plt.figure(figsize=(10, 5))
for i, (num_filters, overlap_dict) in enumerate(overlaps_200.items()):
    plt.plot([10, 30, 50], [overlap_dict['best_case']] * 3, markers[i],
             color=colors[i], label=f'Best case: {num_filters}')
    plt.plot([10, 30, 50], [overlap_dict['our_selection']] * 3, markers[i],
             color=colors[i], linestyle='--', label=f'Our Selection: {num_filters}')
    plt.plot([10, 30, 50], [overlap_dict['random_model']] * 3, markers[i],
             color=colors[i], linestyle='-.', label=f'Random Model: {num_filters}')
    plt.plot([10, 30, 50], [overlap_dict['worst_case']] * 3, markers[i],
             color=colors[i], linestyle=':', label=f'Worst Case: {num_filters}')

plt.yticks(np.arange(4, 5.6, step=0.1))
plt.xticks([10, 30, 50])
plt.xlabel('Number of Filters')
plt.ylabel('Average Overlap')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the layout
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt.savefig('Figures/plot_200_nodes_diff_num_filters.eps',
            format='eps', bbox_inches="tight")
plt.close()
