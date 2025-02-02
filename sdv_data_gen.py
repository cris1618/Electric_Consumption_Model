import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
import os


# Define initial data randomly
data = {
    "Season": np.random.choice(["Winter", "Spring", "Summer", "Fall"], 1000),
    "Size": np.random.randint(500, 4000, 1000),
    "Occupants": np.random.randint(1, 6, 1000),
    "Heating Type": np.random.choice(["Electric", "Gas", "Solar", "None"], 1000),
    "Cooling Type": np.random.choice(["Central AC", "Fans", "None"], 1000),
    "KWh Consumption": np.random.randint(200, 2000, 1000),
}

# Create the DataFrame
df = pd.DataFrame(data)
print(df.head())

# Generate metadata from df
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Visualize metadata 
metadata.visualize()
quit()
# Train the model with the random data 
model = GaussianCopulaSynthesizer(metadata)
model.fit(df)

# Generate sythetic data
s_data = model.sample(num_rows=1000)
print(s_data.head())
