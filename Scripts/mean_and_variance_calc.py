import pandas as pd

# Define all the columns (including non-numerical ones)
# COLUMNS = [
#     'station_name', 'report_timestamp', 'time_since_launch', 'report_id', 'longitude', 'latitude',
#     'height_of_station_above_sea_level', 'altitude', 'altitude_total_uncertainty', 'air_pressure',
#     'air_pressure_total_uncertainty', 'air_temperature', 'air_temperature_total_uncertainty',
#     'air_temperature_random_uncertainty', 'air_temperature_systematic_uncertainty',
#     'air_temperature_post_processing_radiation_correction', 'relative_humidity',
#     'relative_humidity_total_uncertainty', 'relative_humidity_random_uncertainty',
#     'relative_humidity_systematic_uncertainty', 'relative_humidity_post_processing_radiation_correction',
#     'wind_speed', 'wind_speed_total_uncertainty', 'wind_from_direction', 'wind_from_direction_total_uncertainty',
#     'eastward_wind_component', 'northward_wind_component', 'shortwave_radiation', 
#     'shortwave_radiation_total_uncertainty', 'vertical_speed_of_radiosonde', 'geopotential_height', 
#     'water_vapor_volume_mixing_ratio', 'frost_point_temperature', 'relative_humidity_effective_vertical_resolution'
# ]

# Define the USED columns only
COLUMNS = [
    'air_pressure', 'air_temperature', 'relative_humidity', 'wind_speed', 'water_vapor_volume_mixing_ratio',
    'air_pressure_total_uncertainty', 'air_temperature_total_uncertainty', 'air_temperature_random_uncertainty',
    'relative_humidity_total_uncertainty', 'relative_humidity_systematic_uncertainty', 'relative_humidity_post_processing_radiation_correction',
    'wind_speed_total_uncertainty', 'wind_from_direction', 'wind_from_direction_total_uncertainty', 'eastward_wind_component', 'northward_wind_component',
    'shortwave_radiation_total_uncertainty', 'vertical_speed_of_radiosonde', 'geopotential_height', 'frost_point_temperature'
]

# Load the dataset
data_df = pd.read_csv('../Cop_dataset.csv', usecols=COLUMNS)

# Drop rows where all values are NaN
data_df = data_df.dropna(how='all')

# Automatically select only the numerical columns for mean and variance calculation
numerical_columns = data_df.select_dtypes(include=['float64', 'int64']).columns

# Fill missing values in numerical columns with the column mean
data_df[numerical_columns] = data_df[numerical_columns].fillna(data_df[numerical_columns].mean())

# Compute mean and variance for the numerical columns
mean_values = data_df[numerical_columns].mean()
variance_values = data_df[numerical_columns].var()

# Create a DataFrame to store the mean and variance
results = pd.DataFrame({
    'Mean': mean_values,
    'Variance': variance_values
})

# Save the results to an Excel file
output_file = 'used_dims_statistics_results.xlsx'
results.to_excel(output_file, engine='xlsxwriter')

print(f"Mean and variance statistics have been saved to {output_file}.")
