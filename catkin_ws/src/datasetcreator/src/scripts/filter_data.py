import pandas as pd

# Load the CSV file
df = pd.read_csv("catkin_ws/src/datasetcreator/src/data.csv")

# Ensure 'velocity' and 'position' columns are lists of floats, handling cases where entries might be floats directly
#df['velocity'] = df['velocity'].apply(lambda x: list(map(float, str(x).strip("[]").split(", "))) if isinstance(x, str) else x)
df['position'] = df['position'].apply(lambda x: list(map(float, str(x).strip("[]").split(", "))) if isinstance(x, str) else x)

# Filter out rows where all velocity components are zero and x position is within ±10.005
filtered_df = df[
    #(df['velocity'].apply(lambda v: not (v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0))) &
    (df['position'].apply(lambda p: abs(p[0]) <= 10.005))
]

# Save or view the filtered DataFrame
filtered_df.to_csv('catkin_ws/src/datasetcreator/src/filtered_data.csv', index=False)
#print(filtered_df)