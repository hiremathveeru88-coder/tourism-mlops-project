
import pandas as pd

# Load dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
