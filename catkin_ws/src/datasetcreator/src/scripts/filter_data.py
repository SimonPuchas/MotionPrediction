import pandas as pd

# Load the CSV file
df = pd.read_csv("src/trajectory_recorder/src/trajectory_data.csv")

# Filter out rows where "your_column" is 0
df_filtered = df[df["vel_x"] != 0]

# Save the filtered data back to a new CSV (or overwrite the original file)
df_filtered.to_csv("src/trajectory_recorder/src/filtered_data.csv", index=False)