import pandas as pd
import numpy as np


import os

def apply_log_on_one_home(file_name):
    # Define the input and output file paths
    input_file_path = 'Radon dataset/joined_homes/'+file_name
    output_file_path = 'Radon dataset/joined_homes_log/'+file_name

    # Define the name of the column to apply the transformation to
    column_name = 'val1h'

    # Load the input CSV file into a pandas DataFrame
    df = pd.read_csv(input_file_path)

    # Apply the logarithmic transformation to the specified column
    df[column_name] = np.log(df[column_name] / 10 + 1) / np.log(1.025)

    # Save the transformed DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

# Directory containing the files
directory = 'Radon dataset/joined_homes/'

# Loop through the files in the directory
for filename in os.listdir(directory):
    print(filename)
    # Call the log function with the filename
    apply_log_on_one_home(filename)


