import pandas as pd
import os

# Read the training data in chunks to avoid memory issues
input_file = "data/raw/train.csv"
output_file = "data/raw/drift_test.csv"

print("Reading train.csv and filtering for dates >= 2016-01-01...")

# Read in chunks and filter
chunks = []
for chunk in pd.read_csv(input_file, chunksize=50000):
    chunk['date'] = pd.to_datetime(chunk['date'])
    drift_chunk = chunk[chunk['date'] >= '2016-01-01']
    if not drift_chunk.empty:
        chunks.append(drift_chunk)

if chunks:
    drift_data = pd.concat(chunks, ignore_index=True)
    drift_data.to_csv(output_file, index=False)
    print(f"✅ Created {output_file}")
    print(f"   Rows: {len(drift_data)}")
    print(f"   Date range: {drift_data['date'].min()} to {drift_data['date'].max()}")
    print(f"\nNow upload this file to the Drift & Retrain page for drift detection!")
else:
    print("❌ No data found for dates >= 2016-01-01")
