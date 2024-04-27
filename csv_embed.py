import pandas as pd

# Read the CSV file into a dataframe
df = pd.read_csv('data/output.csv')

# Create a new dataframe with only the desired columns
new_df = df[['Vehicle_Title', 'Review']]

# Print the new dataframe
new_df
