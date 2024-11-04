import pandas as pd

# Load the CSV file
df = pd.read_csv("catkin_ws/src/datasetcreator/src/data.csv")

# Filter out rows where "your_column" is 0
#df_filtered = df[df["vel_x"] != 0]
df_filtered = df[~((df["time"] >= 0.177) & (df["time"] <= 0.323))]
df_filtered = df_filtered[(df_filtered["x"] >= 0) & (df_filtered["x"] < 10.002)]


# Save the filtered data back to a new CSV (or overwrite the original file)
df_filtered.to_csv("catkin_ws/src/datasetcreator/src/filtered_data.csv", index=False)