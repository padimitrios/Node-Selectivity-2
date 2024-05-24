import simpy
import random
import math
import numpy as np
import time
import scipy.stats as stats
import statsmodels.api as sm
import pandas as pd

# make it read the packet number, number of vectors, nuber of filters from a csv

# Simulation constants
PACKET_NUMBER = 16
NUMBER_OF_NODES = 10
NUMBER_OF_NODE_VECTORS = 10
LOWER_BOUND = 100
NUMBER_OF_FILTERS = 5

# Model constants
K = 10

# Model A
A = 1.5
B = 2.5
C = -2.5
THETA = 3

# Model B
A_2 = 5
B_2 = -0.5

# Stats variables
e = 0
t = []
p = 0
q = 0
r = 0
s = 0

# Array containing the data types of the incoming vectors
TYPE_CHOICES = [
    'air_pressure', 'air_temperature', 'relative_humidity', 'wind_speed', 'water_vapor_volume_mixing_ratio'
]


class Node:
    '''
      @DESC: Class that represents a single node instance
      '''

    def __init__(self, env):
        self.env = env
        # This array contains the min and max values of each type
        self.filter_vectors_table = []
        # This array contains the random generated pairs
        self.env_data = []

    def generate_filters(self, env, df, W):
        '''
            @DESC: Generate the min max values for each of the types of data and store them in the array
            @PARAMs: self (obj)   -> The Node object
                     env  (obj)   -> The simpy env clock
                     df   (obj)   -> The DataFrame containing the data
        '''
        for _ in range(W):
            filters_vector = []

            sequential_rows = retrieve_sequential_rows(df, 10)
            print(sequential_rows)
            actual_high_low = []

            for dim in TYPE_CHOICES:
                min_val = sequential_rows[dim].min()
                max_val = sequential_rows[dim].max()

                actual_high_low.append([min_val, max_val])

            artificial_filters = []

            for high_low in actual_high_low:
                min_val, max_val = high_low
                # Assuming normal distribution, calculate mean and standard deviation
                mean = (min_val + max_val) / 2
                std_dev = (max_val - min_val) / 4

                # Generate artificial filters using normal distribution
                artificial_min = max(
                    min_val, np.random.normal(loc=mean, scale=std_dev))
                artificial_max = min(
                    max_val, np.random.normal(loc=mean, scale=std_dev))

                # Ensure artificial min is less than artificial max
                if artificial_min > artificial_max:
                    artificial_min, artificial_max = artificial_max, artificial_min

                artificial_filters.append([artificial_min, artificial_max])

            for choice_index, choice in enumerate(TYPE_CHOICES):
                # Assuming you have the artificial_min and artificial_max values for each choice
                artificial_min = artificial_filters[choice_index][0]
                artificial_max = artificial_filters[choice_index][1]

                # Create Filter object with artificial min and max values
                choice_data = Filter(
                    env, choice_index, artificial_min, artificial_max)

                filters_vector.append(choice_data)

            # Append the filters_vector to filter_vectors_table
            self.filter_vectors_table.append(
                Filter_Vector(env, 0, filters_vector))

    def add_filter(self, filter_data, timestamp):
        """
        @DESC: Add a new filter to the filters_table based on the provided array of pairs.
              If the length exceeds 5, pop the 0th position element.
        @PARAMS: filter_data (list) -> An array containing pairs of min and max values for each dimension.
                timestamp (int) -> The timestamp of the event.
        """
        filters_vector = [Filter(self.env, i, min_val, max_val)
                          for i, (min_val, max_val) in enumerate(filter_data)]
        self.filter_vectors_table.append(
            Filter_Vector(self.env, timestamp, filters_vector))

        if len(self.filter_vectors_table) > NUMBER_OF_FILTERS:
            self.filter_vectors_table.pop(0)

    def generate_valid_vector(self):
        '''
            @DESC: Generate a valid vector containing random values within the specified min-max ranges for each data type defined in filters_table.
            @PARAMs: self (obj) -> The Node object.
            @RET: vector (list) -> A list containing random values within the specified ranges.
            '''

        vector = []

        # generate a valid vector based on the last filter of the node
        for filter in self.filter_vectors_table[-1].filters:
            min_val = filter.min_val
            max_val = filter.max_val

            random_value = random.uniform(min_val, max_val)
            vector.append(random_value)
        return vector

    def add_additional_filters(self, num_additional_filters):
        """
        @DESC: Add more filters to the filter_vectors_table based on the first filter instance.
               Each dimension in the new filters has a 10% difference from the first filter,
               and there's a 10% chance for each dimension to be totally new.
        @PARAMS: num_additional_filters (int) -> The number of additional filters to add.
        """
        if not self.filter_vectors_table:
            return

        # Get the first filter instance
        base_filter = self.filter_vectors_table[0].filters

        for _ in range(num_additional_filters):
            new_filter = []
            for dim_index, base_dim in enumerate(base_filter):
                # Determine if the dimension is totally new with a 10% chance
                is_new_dimension = random.random() < 0.1

                if is_new_dimension:
                    # If the dimension is totally new, generate a completely new range
                    new_min_val = random.randint(LOWER_BOUND, 500)
                    new_max_val = new_min_val + LOWER_BOUND
                else:
                    # Otherwise, create a new range with a 10% difference from the first filter
                    percent_difference = random.uniform(0.9, 1.1)
                    new_min_val = int(base_dim.min_val * percent_difference)
                    new_max_val = int(base_dim.max_val * percent_difference)

                new_filter.append(
                    Filter(self.env, dim_index, new_min_val, new_max_val))

            # Add the new filter to the filter_vectors_table
            timestamp = self.env.now
            self.filter_vectors_table.append(
                Filter_Vector(self.env, timestamp, new_filter))

            # If the length exceeds 5, pop the 0th position element
            if len(self.filter_vectors_table) > NUMBER_OF_FILTERS:
                self.filter_vectors_table.pop(0)

    def populate_env_data(self, num_vectors, df):
        '''
            @DESC: Populate the env_data list with a specified number of random vectors.
            @PARAMs: self (obj)       -> The Node object.
                    num_vectors (int) -> The number of random vectors to generate and append to env_data.
        '''

        random_row = df.sample(n=1)
        random_multivariable_vector = []

        for _ in range(num_vectors):
            for dim in TYPE_CHOICES:
                # Append the value of the attribute for the random row to the list
                random_multivariable_vector.append(random_row[dim].values[0])

            self.env_data.append(random_multivariable_vector)
            random_multivariable_vector = []

    def calculate_mean_std(self):
        '''
            @DESC:   Calculate the mean and standard deviation of the values in env_data.
                    Returns separate vectors for mean and standard deviation, each with
                    the same dimensions as the data.
            @RET:    Tuple (mean_vector, std_dev_vector), where mean_vector and std_dev_vector
                    are arrays containing the mean and standard deviation for each dimension
                    of the data.
            '''

        # Check if there's data in the env_data
        if not self.env_data:
            return [0 for _ in TYPE_CHOICES]  # Return None if there's no data

        # Transpose the data
        data = np.array(self.env_data).T

        # Calculate mean and std for each column (dimension)
        mean_vector = np.mean(data, axis=1)
        # std_dev_vector = np.std(data, axis=1)
        # print(mean_vector)
        return mean_vector

    def is_within_filters(self, vector):
        """
            @DESC:   Checks if the vector_value is within the min-max ranges specified
                    in the last/most recent filter in the filter_vectors_table.
            @PARAMS: vector_value (list) -> The vector value to check.
            @RET:    int. Returns 0 if the vector is outside the specified ranges for any
                    dimension, returns 1 if all dimensions are within the specified ranges.
            """

        if not vector:
            return 0  # Return 0 if vector_value is empty

        for filter in self.filter_vectors_table[-1].filters:
            if not (filter.min_val <= vector[filter.data_type] <= filter.max_val):
                return 0

        # If all dimensions with valid ranges are within the specified ranges, append the node and return 1
        self.env_data.append(vector)
        return 1

    def add_future_filters(self, prediction_interval):
        pass


class Filter_Vector:
    '''
      @DESC: Class that represents a filter vector at a specific timestamp
      '''

    def __init__(self, env, timestamp, filters):
        self.env = env
        self.timestamp = timestamp
        self.filters = filters


class Filter:
    '''
      @DESC: Class that represents the different types of data that can occupy the nodes
      '''

    def __init__(self, env, data_type, min_val, max_val):
        self.env = env
        self.data_type = data_type
        self.min_val = min_val
        self.max_val = max_val
        self.pair = [min_val, max_val]


def retrieve_sequential_rows(df, quantity, start_index=None):
    '''
    @DESC: Retrieve a sequence of 10 rows from the DataFrame starting from the specified index.
    @PARAMS: df (DataFrame) -> The DataFrame to retrieve rows from.
                quantity (int) -> The number of rows to retrieve.
                start_index (int) -> The starting index for the sequence of rows.
    @RET: DataFrame -> The sequence of 10 rows starting from the specified index.
    '''
    if start_index is None:
        # Choose a random starting index
        start_index = random.randint(0, len(df) - quantity)

    end_index = start_index + 10  # Calculate the end index

    return df.iloc[start_index:end_index]


def generate_random_multivariable_vector(df):
    '''
    @DESC: Generate a random vector to be filtered
    @RET:  (list)
    '''
    # Select a random row from the DataFrame
    random_row = df.sample(n=1)

    random_multivariable_vector = []

    # Iterate over each attribute in TYPE_CHOICES
    for dim in TYPE_CHOICES:
        random_multivariable_vector.append(random_row[dim].values[0])

    return random_multivariable_vector


def random_node_choice(nodes):
    '''
      @DESC:   Randomly selects a node from the list of nodes and returns both
               the selected node and its index.
      @RET:    Tuple (selected_node, index), where selected_node is a Node instance
               and index is its index in the list of nodes.
      '''

    if not nodes:
        return None, None  # Return None if there are no nodes

    index = random.randint(0, len(nodes) - 1)
    selected_node = nodes[index]

    return selected_node, index


def calculate_min_max_values(random_vector_array):
    min_max_values_array = []

    # Transpose the input array
    transposed_array = np.array(random_vector_array)

    # Find min and max values for each dimension
    min_values = np.min(transposed_array, axis=0)
    max_values = np.max(transposed_array, axis=0)

    min_max_values_array.append(min_values)
    min_max_values_array.append(max_values)

    return min_max_values_array


def update_stats(node, vector):
    '''
      @DESC:   Update the variables accounted for the statistics of the sim
      @PARAMS: node (obj)   -> The node obj
              vector_value  -> the random generated vector
      '''

    global p, q, r, s

    dimensions = len(vector)
    values_within = 0

    for filter in node.filter_vectors_table[-1].filters:
        min_val = filter.min_val
        max_val = filter.max_val

        element_value = vector[filter.data_type]
        if min_val <= element_value <= max_val:
            values_within += 1

    if values_within == dimensions:
        p += 1
    elif values_within >= dimensions / 2:
        r += 1
    elif (values_within < dimensions / 2) and values_within > 0:
        s += 1
    else:
        q += 1


def calculate_current_overlaps(env, nodes, min_max_values_array):
    """
    @DESC:   Calculate the current overlap between the filters in nodes.filters_table and
             the corresponding elements in the random_vector for each node.
    @PARAMS: env (obj)               -> The SimPy environment.
             nodes (list of Node)    -> List of Node objects.
             random_vector (list)     -> The N-length array for calculating overlap.
    @RET:    list of tuples -> Each tuple contains the node index and the corresponding total overlap.
    """

    # Ensure that the nodes list is not empty
    if not nodes:
        return []

    # Initialize list to store tuples of node index and total overlap
    node_overlaps = []

    # Iterate over each node and calculate overlap with its filters
    for idx, node in enumerate(nodes):
        node_overlap = calculate_node_overlap(node.filter_vectors_table,
                                              min_max_values_array)
        node_overlaps.append((idx, node_overlap))

    return node_overlaps


def calculate_node_overlap(filter_vectors_table, min_max_values_array):
    """
    @DESC:   Calculate the overlap between the filters in filters_table and the corresponding
             elements in the random_vector for a specific node.
    @PARAMS: filters_table (list)   -> The filters table for a specific node.
             random_vector (list)    -> The N-length array for calculating overlap.
    @RET:    float -> The calculated overlap value for the node.
    """

    # Initialize overlap value for the node
    node_overlap = 0.0
    node_overlaps = []

    for filter_vector in filter_vectors_table:

        filter_overlap = []
        # Iterate over each filter and calculate overlap with the corresponding element in random_vector
        for i, filter_pair in enumerate(filter_vector.filters):
            min_filter, max_filter = filter_pair.min_val, filter_pair.max_val
            min_vector_val, max_vector_val = min_max_values_array[0][i], min_max_values_array[1][i]
            element_overlap = calculate_element_overlap(
                min_filter, max_filter, min_vector_val, max_vector_val)
            filter_overlap.append(element_overlap)
            # node_overlap += element_overlap

        node_overlaps.append(filter_overlap)

    average_node_overlap = np.mean(node_overlaps, axis=0)

    return average_node_overlap


def calculate_element_overlap(min_filter, max_filter, min_vector_val, max_vector_val):
    # Ensure that the min_val and max_val are valid
    if max(min_filter, min_vector_val) < min(max_filter, max_vector_val):
        intersection_tuple = [
            max(min_filter, min_vector_val),
            min(max_filter, max_vector_val)
        ]
        intersection = abs(intersection_tuple[0] - intersection_tuple[1])

        overlap = 1 - intersection / \
            (min(abs(min_vector_val - max_vector_val), abs(min_filter - max_filter)))
    else:
        overlap = 0.0  # No overlap

    return overlap


def calculate_future_intervals(nodes):
    # Set the desired probability
    p = 0.90

    # Ensure that the nodes list is not empty
    if not nodes:
        return []

    result_dict = {}

    # Iterate over each node and calculate overlap with its filters
    for index, node in enumerate(nodes):
        all_filter_vectors = []
        for filter_vector in node.filter_vectors_table:
            complete_filter_vector = []
            for filter in filter_vector.filters:
                complete_filter_vector.append(filter.pair)
            all_filter_vectors.append(complete_filter_vector)

        prediction_intervals = []

        # Iterate through each pair of vectors
        for dimension in range(len(TYPE_CHOICES)):
            # Extract the values for the current dimension
            values = [all_filter_vectors[i][dimension]
                      for i in range(len(all_filter_vectors))]

            # Flatten the list of values
            values_flat = np.concatenate(values)

            # Calculate sample mean and standard deviation
            sample_mean = np.mean(values_flat)
            # ddof=1 for sample standard deviation
            sample_std = np.std(values_flat, ddof=1)

            # Ancillary statistic
            ancillary_statistic = sample_std / \
                np.sqrt(1 + 1 / len(values_flat))

            # Calculate the t-score based on the confidence level and degrees of freedom
            t_score = stats.t.ppf((1 + p) / 2, df=len(values_flat) - 1)

            # Calculate the standard error of the mean
            standard_error = sample_std / np.sqrt(len(values_flat))

            # Calculate the prediction interval
            lower_bound = sample_mean - t_score * ancillary_statistic
            upper_bound = sample_mean + t_score * ancillary_statistic

            prediction_intervals.append([lower_bound, upper_bound])

        result_dict[index] = prediction_intervals

    return result_dict


def calculate_quantile_regression_for_nodes(nodes, min_max_array):
    """
    @DESC: Calculate quantile regression for nodes.
    @PARAMS: nodes (list) -> List of nodes containing filter vectors.
             min_max_array (array-like) -> Array with min and max values for prediction.
    @RET: dict -> Results of quantile regression for each node and dimension.
                 {
                     node_index: {
                         dimension_index: [
                             {
                                 'quantile_params': Quantile regression parameters,
                                 'low_res': Predicted values for low dimension,
                                 'high_res': Predicted values for high dimension
                             },
                             ...
                         ],
                         ...
                     },
                     ...
                 }
    """
    full_node_results = {}

    for node_index, node in enumerate(nodes):
        filter_vectors_table = node.filter_vectors_table
        node_results = {}

        for dim_index, dimension in enumerate(TYPE_CHOICES):
            dimension_results = []
            dimension_data = []

            # Extract data for the current dimension
            for filter_vector in filter_vectors_table:
                for filter in filter_vector.filters:
                    if filter.data_type == dim_index:
                        dimension_data.append(filter.pair)
                        break

            data_array = np.array(dimension_data)

            # Extract independent variable (X) and dependent variable (Y_low, Y_high)
            Y_low_dim = data_array[:, 0]
            Y_high_dim = data_array[:, 1]

            # Apply quantile regression for low dimension
            quantile_low = sm.QuantReg(
                Y_low_dim, sm.add_constant(Y_low_dim)).fit(q=0.5)
            low_res = quantile_low.predict([1, min_max_array[0][dim_index]])

            # Apply quantile regression for high dimension
            quantile_high = sm.QuantReg(
                Y_high_dim, sm.add_constant(Y_low_dim)).fit(q=0.5)
            high_res = quantile_high.predict([1, min_max_array[1][dim_index]])

            results_dict = {
                'quantile_low_params': quantile_low.params,
                'quantile_high_params': quantile_high.params,
                'low_res': low_res,
                'high_res': high_res,
                # You can add more information here if needed
            }

            dimension_results.append(results_dict)
            node_results[dim_index] = dimension_results

        full_node_results[node_index] = node_results

    return full_node_results


def aggregate_statistical_results(overlap_values, future_intervals):

    # Initialize an empty dictionary to store aggregated results
    aggregated_results = {}

    # Iterate over the data
    for key, values in future_intervals.items():
        total_avg = 0
        # Iterate over each sublist in the values
        for sublist in values:
            # Calculate the average of the high and low values in the sublist
            sublist_avg = sum(sublist) / len(sublist)
            # Add the average of the sublist to the total average for the key
            total_avg += sublist_avg
        # Store the total average for the key in the aggregated_results dictionary
        aggregated_results[key] = total_avg

    # print(aggregated_results)

    # Iterate over the given array and calculate the sum for each array
    for key, array in overlap_values:
        array_sum = sum(array)
        # Add the sum to the current value for the key
        aggregated_results[key] += array_sum

    # print(aggregated_results)
    return aggregated_results


def aggregate_ml_overlaps_with_st_overlaps(ml_overlaps, st_overlaps):
    pass


def calculate_intervals_overlap(future_intervals, min_max_values_array):
    """
    @DESC:   Calculate the overlap between future intervals and the corresponding
             elements in the min_max_values_array.
    @PARAMS: future_intervals (dict) -> Dictionary containing future intervals for each node.
             min_max_values_array (list) -> The N-length array for calculating overlap.
    @RET:    list of lists -> Each inner list contains the overlap values for one interval.
    """
    interval_overlaps = []

    for node_idx, intervals in future_intervals.items():
        node_overlaps = []
        for interval in intervals:
            interval_overlap = calculate_interval_overlap(
                interval, min_max_values_array)
            node_overlaps.append(interval_overlap)
        interval_overlaps.append(node_overlaps)

    return interval_overlaps


def calculate_interval_overlap(interval, min_max_values_array):
    """
    @DESC:   Calculate the overlap between an interval and the min_max_values_array.
    @PARAMS: interval (list) -> The interval to calculate overlap for.
             min_max_values_array (list) -> The N-length array for calculating overlap.
    @RET:    float -> The calculated overlap value for the interval.
    """
    overlap_values = []

    for min_val, max_val in zip(*min_max_values_array):
        overlap = calculate_element_overlap(min_val, max_val, *interval)
        overlap_values.append(overlap)

    return np.mean(overlap_values)


def calculate_element_overlap(min_val, max_val, interval_start, interval_end):
    """
    @DESC:   Calculate the overlap between an element (min_val, max_val) and an interval.
    @PARAMS: min_val (float) -> The minimum value of the element.
             max_val (float) -> The maximum value of the element.
             interval_start (float) -> The start of the interval.
             interval_end (float) -> The end of the interval.
    @RET:    float -> The calculated overlap value for the element and interval.
    """
    if max(min_val, interval_start) < min(max_val, interval_end):
        intersection = min(max_val, interval_end) - \
            max(min_val, interval_start)
        overlap = intersection / (max_val - min_val)
    else:
        overlap = 0.0  # No overlap

    return overlap


def calculate_quantile_results_overlap(quant_results, min_max_values_array):
    """
    @DESC:   Calculate the overlap between quantile results and the corresponding
             elements in the min_max_values_array.
    @PARAMS: quant_results (dict) -> Dictionary containing quantile results for each node.
             min_max_values_array (list) -> The N-length array for calculating overlap.
    @RET:    list of lists -> Each inner list contains the overlap values for one node's quantile results.
    """
    overlaps = []

    for node_idx, node_data in quant_results.items():
        node_overlaps = []
        for interval_idx, interval_data in node_data.items():
            interval = interval_data[0]['low_res'][0], interval_data[0]['high_res'][0]
            overlap = calculate_interval_overlap(
                interval, min_max_values_array)
            node_overlaps.append(overlap)
        overlaps.append(node_overlaps)

    return overlaps


def calculate_interval_overlap(interval, min_max_values_array):
    """
    @DESC:   Calculate the overlap between an interval and the min_max_values_array.
    @PARAMS: interval (tuple) -> The interval to calculate overlap for.
             min_max_values_array (list) -> The N-length array for calculating overlap.
    @RET:    list -> The calculated overlap values for the interval.
    """
    overlap_values = []

    for min_val, max_val in zip(*min_max_values_array):
        overlap = calculate_element_overlap(min_val, max_val, *interval)
        overlap_values.append(overlap)

    return np.mean(overlap_values)


def calculate_element_overlap(min_val, max_val, interval_start, interval_end):
    """
    @DESC:   Calculate the overlap between an element (min_val, max_val) and an interval.
    @PARAMS: min_val (float) -> The minimum value of the element.
             max_val (float) -> The maximum value of the element.
             interval_start (float) -> The start of the interval.
             interval_end (float) -> The end of the interval.
    @RET:    float -> The calculated overlap value for the element and interval.
    """
    if max(min_val, interval_start) < min(max_val, interval_end):
        intersection = min(max_val, interval_end) - \
            max(min_val, interval_start)
        overlap = intersection / (max_val - min_val)
    else:
        overlap = 0.0  # No overlap

    return overlap


def statistical_overlap_calculation(node_overlaps, future_overlaps):
    """
      @DESC:   Calculate the statistical overlap between corresponding rows of node overlaps and future overlaps.
      @PARAMS: node_overlaps (list of tuples) -> List containing tuples where each tuple contains an index and an array of overlap values.
               future_overlaps (list of lists) -> List of lists where each inner list represents overlap values for future intervals.
      @RET:    dict -> A dictionary where the index from node_overlaps serves as the key and the symmetric average overlap serves as the value.
    """

    statistical_overlaps = {}
    for i in range(len(future_overlaps)):
        future_row_sum = np.sum(future_overlaps[i])
        node_row_sum = np.sum(node_overlaps[i][1])
        symmetric_avg = (future_row_sum + node_row_sum) / 2
        statistical_overlaps[node_overlaps[i][0]] = symmetric_avg

    return statistical_overlaps


def ml_overlap_aggregation(ml_overlaps):
    """
    @DESC:   Calculate the statistical overlap for each row in the future overlaps table.
    @PARAMS: future_overlaps (list of lists) -> List of lists where each inner list represents overlap values for future intervals.
    @RET:    dict -> A dictionary where the index serves as the key and the symmetric average overlap serves as the value.
    """
    # print(ml_overlaps)
    ml_overlaps_aggr = {}
    for i in range(len(ml_overlaps)):
        row_sum = sum(ml_overlaps[i])
        symmetric_avg = row_sum / len(TYPE_CHOICES)
        ml_overlaps_aggr[i] = symmetric_avg
    return ml_overlaps_aggr


def aggregate_overlaps(statistical_overlaps, ml_overlaps):
    """
    @DESC:   Aggregate two dictionaries by calculating the average for each key.
    @PARAMS: statistical_overlaps (dict) -> Dictionary containing statistical overlaps.
             ml_overlaps (dict) -> Dictionary containing overlaps from machine learning.
    @RET:    dict -> A dictionary where the index serves as the key and the symmetric average overlap serves as the value.
    """

    aggregated_overlaps = {}

    for key in statistical_overlaps.keys():
        aggregated_overlaps[key] = (
            statistical_overlaps[key] + ml_overlaps[key]) / 2

    sorted_overlaps = dict(
        sorted(aggregated_overlaps.items(), key=lambda item: item[1]))

    return sorted_overlaps


def data_selectivity(env, nodes, PACKET_NUMBER, PACKET_THRESHOLD, data_df):
    '''
      @DESC:    Simulates the selectivity of data in edge computing environments
      @PARAMS:  env (simpy.Environment) -> the simulation environment
                nodes (list) -> the list of nodes to simulate
                PACKET_NUMBER (int) -> the number of packets to simulate
      '''

    global e

    # array that will store an interval of random vectors
    random_vector_array = []

    for _ in range(PACKET_NUMBER):

        start_time = time.time()

        random_node, random_nodes_index = random_node_choice(nodes)
        candidate_node, candidate_nodes_index = random_node_choice(nodes)
        # # duplicate check
        # while True:
        #   candidate_node, candidate_nodes_index = random_node_choice(nodes)
        #   if candidate_nodes_index != random_nodes_index:
        #     break

        random_vector = random_node.calculate_mean_std()

        # reset interval table
        if len(random_vector_array) < PACKET_THRESHOLD:
            random_vector_array.append(random_vector)
            continue
        else:

            min_max_values_array = calculate_min_max_values(
                random_vector_array)

            # calculate the overlap based on the min-max values meaning there are 4 values to calculate the overlap

            node_overlaps = calculate_current_overlaps(
                env, nodes, min_max_values_array)
            # print(min_max_values_array)
            # print(node_overlaps)

            # print(node_overlaps)
            # sorted_overlaps = sorted(node_overlaps, key=lambda x: np.sum(x[1]))
            # closest_node_index = sorted_overlaps[0][0]
            future_intervals = calculate_future_intervals(nodes)
            future_intervals_overlaps = calculate_intervals_overlap(
                future_intervals, min_max_values_array)

            # print(future_intervals_overlaps)

            #  calculate the future nodes overlaps

            # aggr_results = aggregate_statistical_results(node_overlaps, future_intervals)
            # print(aggr_results)
            quantile_reg_results = calculate_quantile_regression_for_nodes(
                nodes, min_max_values_array)
            quant_overlaps = calculate_quantile_results_overlap(
                quantile_reg_results, min_max_values_array)
            # print(quant_overlaps)
            # print(statistical_overlap)
            ml_overlaps = ml_overlap_aggregation(quant_overlaps)
            statistical_overlap = statistical_overlap_calculation(
                node_overlaps, future_intervals_overlaps)
            # print(ml_overlaps)
            aggregated_result = aggregate_overlaps(
                statistical_overlap, ml_overlaps)
            print(aggregated_result)
            # print(future_intervals)
            # print(quantile_reg_results)

            # aggregated_overlaps, aggregated_intervals, aggregated_quantile_coefs = aggregate_results(nodes, node_overlaps, future_intervals, quantile_reg_results)
            # print(future_intervals)
            # print(quantile_reg_results)
            break
            # print()
            # print("Aggregated Overlaps:", aggregated_overlaps)
            # print("Aggregated Future Intervals:", aggregated_intervals)
            # print("Aggregated Quantile Regression Coefficients:", aggregated_quantile_coefs)

            # if closest_node_index == candidate_nodes_index:
            #   print('match found on candidate node')
            #   end_time = time.time()
            #   e += 1

            #   # update the filters table
            #   add_future_filters(random_vector, nodes)

            #   elapsed_time = end_time - start_time
            #   t.append(elapsed_time)
            #   update_stats(candidate_node, random_vector)
            #   print(f"Elapsed time: {elapsed_time} seconds")
            # else:
            #   #select closest node and add the vector
            #   # update the filters table
            #   add_future_filters(random_vector, nodes)
            #   # add the future filters

            #   end_time = time.time()

            #   # Calculate the elapsed time
            #   elapsed_time = end_time - start_time
            #   t.append(elapsed_time)
            #   print(f"Elapsed time: {elapsed_time} seconds")
            random_vector_array = []
            random_vector_array.append(random_vector)

    # print(t)
    # print("Nodes fully matched:", p)
    # print("Nodes fully unmatched:", q)
    # print("Nodes partially matched:", r)
    # print("Nodes partially unmatched:", s)
    # print("Nodes went to candidate:", e)
    yield env.timeout(1)


if __name__ == '__main__':
    constants_df = constants_df = pd.read_csv('constants.csv')
    data_df = pd.read_csv(
        'Cop_dataset.csv', usecols=TYPE_CHOICES)
    data_df = data_df.dropna(how='all')

    W = 10

    for index, row in constants_df.iterrows():
        # Update the constants for the current row
        PACKET_NUMBER = row['PACKET_NUMBER']
        NUMBER_OF_NODES = row['NUMBER_OF_NODES']
        NUMBER_OF_NODE_VECTORS = row['NUMBER_OF_NODE_VECTORS']
        LOWER_BOUND = row['LOWER_BOUND']
        NUMBER_OF_FILTERS = row['NUMBER_OF_FILTERS']
        PACKET_THRESHOLD = row['PACKET_THRESHOLD']
        K = row['K']

        # Print the current set of constants
        print(f"Running simulation with constants from row {index + 1}:")
        print(f"PACKET_NUMBER: {PACKET_NUMBER}")
        print(f"NUMBER_OF_NODES: {NUMBER_OF_NODES}")
        print(f"NUMBER_OF_NODE_VECTORS: {NUMBER_OF_NODE_VECTORS}")
        print(f"LOWER_BOUND: {LOWER_BOUND}")
        print(f"NUMBER_OF_FILTERS: {NUMBER_OF_FILTERS}")
        print(f"PACKET_THRESHOLD: {PACKET_THRESHOLD}")
        print(f"K: {K}")

        env = simpy.Environment()
        nodes = []

        # generate the nodes and populate them with data
        for i in range(NUMBER_OF_NODES):
            node = Node(env)
            node.generate_filters(env, data_df, W)
            node.populate_env_data(NUMBER_OF_NODE_VECTORS, data_df)
            # node.add_additional_filters(num_additional_filters=4)
            # for filter in node.filter_vectors_table:
            #   for index, f in enumerate(filter.filters):
            #     print(f.pair, index)
            nodes.append(node)

        # Run the simulation for PACKET_NUMBER packets
        env.process(data_selectivity(
            env, nodes, PACKET_NUMBER, PACKET_THRESHOLD, data_df))
        env.run(until=PACKET_NUMBER)
