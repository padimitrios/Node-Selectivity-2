import pandas as pd
import numpy as np
import ast
import random

# Function to parse a stringified list
def parse_list(array_string):
    try:
        return ast.literal_eval(array_string)
    except (ValueError, SyntaxError):
        return []

# Step 1: Load the dataset
data = pd.read_csv('greedy_rec.csv')  # replace with your actual CSV file

# Step 2: Parse 'our_model_selections' and 'random_model' as lists
data['our_model_selections'] = data['our_model_selections'].apply(parse_list)
data['random_model'] = data['random_model'].apply(parse_list)

# Step 3: Reconstruct 'greedy_selection' by randomly picking from 'our_model_selections' or 'random_model'
def reconstruct_greedy(our_model_selections, random_model):
    if len(our_model_selections) != len(random_model):
        return np.nan  # handle inconsistent lengths
    return [random.choice([our_model_selections[i], random_model[i]]) for i in range(len(our_model_selections))]

data['greedy_selection'] = data.apply(
    lambda row: reconstruct_greedy(row['our_model_selections'], row['random_model']),
    axis=1
)

# Step 4: Save the modified dataset (optional)
data.to_csv('modified_dataset.csv', index=False)