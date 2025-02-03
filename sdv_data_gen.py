import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot, get_column_pair_plot
import os
import graphviz
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

"""# Define initial data randomly
data = {
    "Month": np.random.choice(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 10000),
    "Season": np.random.choice(["Winter", "Spring", "Summer", "Fall"], 10000),
    "Size": np.random.randint(500, 4000, 10000),
    "Occupants": np.random.randint(1, 6, 10000),
    "Heating Type": np.random.choice(["Electric", "Gas", "Solar", "None"], 10000),
    "Cooling Type": np.random.choice(["Central AC", "Fans", "None"], 10000),
    "KWh Consumption": np.random.randint(200, 2000, 10000),
}"""

# Dataset size
num_samples = 1000

# Creating the data in a way that they are not completely random,
# For example, at a larger home size will correspond higher consumption
# and so on for the other features.

# Generate the house size with a Gaussian distribution
house_size = np.random.normal(loc=2000, scale=600, size=num_samples).astype(int)
house_size = np.clip(house_size, 800, 4000) # Limit the size to realistic scales

# More occupants in larger homes
occupants = np.random.poisson(lam=house_size / 1000 + 2, size=num_samples)
occupants = np.clip(occupants, 1, 8)

# Seasons
seasons = np.random.choice(["Winter", "Spring", "Summer", "Fall"], num_samples)

# Heating type (More likely to be electric in colder months)
heating_type = np.where(
    (seasons == "Winter") & (np.random.rand(num_samples) > 0.3), "Electric",
    np.random.choice(["Gas", "Solar", "None"], num_samples)
)

# Cooling type (AC during the summer)
cooling_type = np.where(
    (seasons == "Summer") & (np.random.rand(num_samples) > 0.4), "Central AC",
    np.random.choice(["Fans", "None"], num_samples)
)

# Energy consumption (must be correlated to the other variables)
energy_consumption = (
    house_size * 0.4 +
    occupants * 50 +
    np.where(heating_type == "Electric", 300, 0) +
    np.where(cooling_type == "Central AC", 200, 0) +
    np.random.randint(-100, 100, num_samples) # Add some noise
).astype(int)
energy_consumption = np.clip(energy_consumption, 200, 3000) # Keep it within reasonable bounderies

# Create the DataFrame
df = pd.DataFrame({
    "Season": seasons,
    "Size": house_size,
    "Occupants": occupants,
    "Heating Type": heating_type,
    "Cooling Type": cooling_type,
    "KWh Consumption": energy_consumption
})
print(df.head())

# Generate metadata from df
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Visualize metadata 
graph = metadata.visualize()
#graph.view()

# Train the model with the random data 
model = GaussianCopulaSynthesizer(metadata)
model.fit(df)

# Generate sythetic data
s_data = model.sample(num_rows=10000)
s_data.to_csv("SyntethicData/synthetic_energy_data.csv", index=False)
print(s_data.head())

# Compare real and synthetic data
diagnostic = run_diagnostic(
    real_data=df,
    synthetic_data=s_data,
    metadata=metadata
)

# Evaluate the data quality of the syntethic data
quality_report = evaluate_quality(
    real_data=df,
    synthetic_data=s_data,
    metadata=metadata
)

# Compare distributions of the "Size" column
fig = get_column_plot(
    real_data=df,
    synthetic_data=s_data,
    column_name="Size",
    metadata=metadata
)

# Comparing correlations between synthetic and real data
fig = get_column_pair_plot(
    real_data=df,
    synthetic_data=s_data,
    column_names=["Size", "KWh Consumption"],
    metadata=metadata
)

fig.show()