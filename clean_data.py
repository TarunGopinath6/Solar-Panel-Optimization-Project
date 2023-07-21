import pandas as pd

data = pd.read_csv("output.csv")
new_data = data[data["SOURCE_KEY"] == "1BY6WEcLGh8j5v7"]
# new_df = df[df["SOURCE_KEY"] == "1BY6WEcLGh8j5v7"]
new_data.to_csv("output_source.csv", index=False)
# # Save the new dataset to a CSV file
# new_df.to_csv("./output_source.csv", index=False)
