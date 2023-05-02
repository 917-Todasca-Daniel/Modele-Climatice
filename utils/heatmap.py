import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.constants import PATH_TO_DATASET

MAX_NUMBER_OF_HOUSES = 10
FIRST_NUMBER_OF_MEASUREMENTS = 10

houses = []
values = []

# For MAX_NUMBER_OF_HOUSES houses read the first FIRST_NUMBER_OF_MEASUREMENTS measurements
i = 0
for filename in os.listdir(PATH_TO_DATASET):
    if i >= MAX_NUMBER_OF_HOUSES:
        break
    i += 1
    houses.append(filename)
    df = pd.read_csv(PATH_TO_DATASET + "/" + filename)

    values.append(df['val1h'].values)

# Cut the values to the first FIRST_NUMBER_OF_MEASUREMENTS measurements
values = [value[:FIRST_NUMBER_OF_MEASUREMENTS] for value in values]

plt.figure(figsize=(len(houses), len(values)))
heat_map = sns.heatmap(values, annot=True)

plt.title("HeatMap Of Radon Values")
plt.ylabel("House")
plt.xlabel("First " + str(FIRST_NUMBER_OF_MEASUREMENTS) + " measurements")
plt.show()
