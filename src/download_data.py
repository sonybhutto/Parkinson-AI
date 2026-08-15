import pandas as pd
import urllib.request
import os

# Create data/raw directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Download the Parkinson's dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
output_path = "data/raw/parkinsons.csv"

# Download the file
print("Downloading dataset...")
urllib.request.urlretrieve(url, output_path)
print(f"Dataset downloaded to {output_path}")

# Quick verification
df = pd.read_csv(output_path)
print(f"\n✅ Dataset shape: {df.shape}")
print(f"✅ Columns: {df.columns.tolist()}")
print(f"\n📊 First 5 rows:")
print(df.head())
print(f"\n📈 Target distribution:")
print(df['status'].value_counts())