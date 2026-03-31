import pandas as pd
import numpy as np
import os
from pathlib import Path

# Get project root directory (works from any working directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Load dataset - using the actual available dataset
df = pd.read_csv(PROJECT_ROOT / "data/raw/Ecommerce_Sales_Prediction_Dataset.csv")

# Convert Date properly (important)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# -------------------------
# Create OLD DATA (Reference baseline)
# -------------------------
df_old = df.copy()

# -------------------------
# Create NEW DATA (with realistic DRIFT)
# -------------------------
df_new = df.copy()

# 1. Increase Discount (major drift)
df_new["Discount"] = (df_new["Discount"] * 1.5).clip(upper=100)

# 2. Change Price pattern (seasonal/market shift)
df_new["Price"] = df_new["Price"] * np.random.uniform(0.9, 1.2, len(df_new))

# 3. Increase Marketing Spend (increased campaign activity)
df_new["Marketing_Spend"] = df_new["Marketing_Spend"] * 1.3

# 4. Shift Product Category distribution (market preference changes)
df_new["Product_Category"] = np.random.choice(
    df_new["Product_Category"].unique(),
    size=len(df_new)
)

# 5. Randomize Customer Segment (behavior change)
df_new["Customer_Segment"] = np.random.choice(
    df_new["Customer_Segment"].unique(),
    size=len(df_new)
)

# 6. Slightly change target (realistic sales impact)
df_new["Units_Sold"] = (df_new["Units_Sold"] * np.random.uniform(0.9, 1.2, len(df_new))).astype(int)

# -------------------------
# Create output directories if they don't exist
# -------------------------
os.makedirs(PROJECT_ROOT / "data/raw", exist_ok=True)
os.makedirs(PROJECT_ROOT / "powerbi", exist_ok=True)

# -------------------------
# Save datasets
# -------------------------
df_old.to_csv(PROJECT_ROOT / "data/raw/old_data.csv", index=False)
df_new.to_csv(PROJECT_ROOT / "data/raw/new_data.csv", index=False)

# -------------------------
# Combine for Power BI
# -------------------------
df_old["Data_Type"] = "Old"
df_new["Data_Type"] = "New"

combined = pd.concat([df_old, df_new], ignore_index=True)

combined.to_csv(PROJECT_ROOT / "powerbi/combined.csv", index=False)

print("✅ Old, New, and Combined datasets created successfully!")
print(f"  - Old data: {PROJECT_ROOT}/data/raw/old_data.csv ({len(df_old)} rows)")
print(f"  - New data: {PROJECT_ROOT}/data/raw/new_data.csv ({len(df_new)} rows)")
print(f"  - Combined: {PROJECT_ROOT}/powerbi/combined.csv ({len(combined)} rows)")
print(f"\n📊 Data drift introduced:")
print(f"  - Discount: +50% increase")
print(f"  - Price: ±10-20% variation")
print(f"  - Marketing Spend: +30% increase")
print(f"  - Product Category: Random distribution")