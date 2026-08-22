import pandas as pd
import glob
import os

# Dataset folder path
folder_path = "datasets"

# Get all CSV files
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Empty list
all_data = []

# Read all CSV files
for file in csv_files:
    print(f"Reading: {file}")
    df = pd.read_csv(file)
    all_data.append(df)

# Merge all datasets
merged_data = pd.concat(all_data, ignore_index=True)

# Save merged dataset
merged_data.to_csv("SriLanka_Tourism_Dataset.csv", index=False)

print("\n✅ All datasets merged successfully!")
print("Total Rows:", len(merged_data))
print("Total Columns:", len(merged_data.columns))

print("\nColumn Names:")
print(merged_data.columns.tolist())