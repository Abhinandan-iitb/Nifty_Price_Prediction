import pandas as pd

# Create an empty list to store DataFrames
dataframes = []

# Loop through each year from 2015 to 2024
for year in range(2015, 2025):
    # Construct the file name based on the year
    file_name = f'NIFTY_Data{year}.csv'
    
    try:
        # Read the CSV file for the given year
        df = pd.read_csv(file_name)
        
        # Append the DataFrame to the list
        dataframes.append(df)
        
        print(f"Successfully read {file_name}")
    except FileNotFoundError:
        print(f"File {file_name} not found, skipping...")

# Concatenate all the DataFrames in the list
merged_df = pd.concat(dataframes, ignore_index=True)

# Display the merged DataFrame
print(merged_df)

# Optionally, save the merged DataFrame to a new CSV file
merged_df.to_csv('Merged_NIFTY_Data.csv', index=False)
