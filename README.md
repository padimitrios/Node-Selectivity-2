# Experimental Results: Model Selection and Performance

This README file provides an overview of the experimental setup and the results of our model selection experiments. The experiments were conducted to evaluate the performance of our model compared to a random model, in terms of the number of selected nodes, minimum overlaps, and time elapsed. The results are categorized based on different experimental constants.

## Experimental Constants

The following constants were used for our experiments:

- **PACKET_NUMBER**: Identifier for the packet of experiments.
- **NUMBER_OF_NODES**: Total number of nodes in the network.
- **NUMBER_OF_FILTERS**: Number of filters applied.
- **PACKET_THRESHOLD**: Threshold value for packet selection.
- **K**: Number of selections made by the model.
- **DIM**: Dimensionality of the node vectors.

### Constants Values

| PACKET_NUMBER | NUMBER_OF_NODES | NUMBER_OF_FILTERS | PACKET_THRESHOLD | K  | DIM |
| ------------- | --------------- | ----------------- | ---------------- | -- | --- |
| 100           | 100             | 10                | 5                | 1  | 5   |
| 100           | 100             | 10                | 5                | 1  | 10  |
| 100           | 100             | 10                | 5                | 1  | 20  |
| 100           | 100             | 10                | 5                | 5  | 5   |
| 100           | 100             | 10                | 5                | 5  | 10  |
| 100           | 100             | 10                | 5                | 5  | 20  |
| 100           | 100             | 10                | 5                | 10 | 5   |
| 100           | 100             | 10                | 5                | 10 | 10  |
| 100           | 100             | 10                | 5                | 10 | 20  |
| 100           | 100             | 10                | 5                | 20 | 5   |
| 100           | 100             | 10                | 5                | 20 | 10  |
| 100           | 100             | 10                | 5                | 20 | 20  |
| 100           | 200             | 10                | 5                | 1  | 5   |
| 100           | 200             | 10                | 5                | 1  | 10  |
| 100           | 200             | 10                | 5                | 1  | 20  |
| 100           | 200             | 10                | 5                | 5  | 5   |
| 100           | 200             | 10                | 5                | 5  | 10  |
| 100           | 200             | 10                | 5                | 5  | 20  |
| 100           | 200             | 10                | 5                | 10 | 5   |
| 100           | 200             | 10                | 5                | 10 | 10  |
| 100           | 200             | 10                | 5                | 10 | 20  |
| 100           | 200             | 10                | 5                | 20 | 5   |
| 100           | 200             | 10                | 5                | 20 | 10  |
| 100           | 200             | 10                | 5                | 20 | 20  |

## Results DataFrame

The results of the experiments are captured in the following DataFrame:

```python
df = pd.DataFrame({
    'our_model_selections': our_model_selections,
    'random_model': random_model,
    'min_overlaps': min_overlaps,
    'time_elapsed': t,
    'dimensions': DIM,
})
```

### Columns Description

- **our_model_selections**: Number of selections made by our model.
- **random_model**: Number of selections made by the random model.
- **min_overlaps**: Minimum number of overlaps between selections.
- **time_elapsed**: Time taken for the model to make selections.
- **dimensions**: Dimensionality of the input vectors used in the experiment.

### Dimensionality Impact

- **DIM**: Different dimensionality values (5, 10, 20) are tested to observe the impact on model performance and computational time. Higher dimensions may increase computational complexity.

### Node and Filter Variations

- **NUMBER_OF_NODES** and **NUMBER_OF_FILTERS**: Varying these parameters helps us understand how the model scales with different network sizes and complexity.
