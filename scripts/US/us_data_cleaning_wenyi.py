import numpy as np
import pandas as pd

# --- 1. Read the data ---

file_path = "american_bankruptcy_dataset.csv"  
df = pd.read_csv(file_path)

# quick inspection
print('=== First 5 rows of the raw data ===')
print(df.head())
print('\n=== Basic information ===')
print(df.info())
print('\n=== Basic descriptive statistcis ===')
print(df.describe(include='all')) 

# --- 2. Define the bankruptcy target variable ---

# Check the unique values in status_label
print("\n=== status_label value counts ===")
print(df["status_label"].value_counts())

# Map status_label to a binary target variable "Bankrupt"
# We assume that:  "alive" -> 0 (non-bankrupt); "failed" -> 1 (bankrupt)

label_map = {
    "alive": 0,
    "failed": 1
}

df["Bankrupt"] = df["status_label"].map(label_map)

print("\n=== Bankrupt (0/1) value counts ===")
print(df["Bankrupt"].value_counts())
print("\n=== Bankrupt class proportions ===")
print(df["Bankrupt"].value_counts(normalize=True))

# --- 3. Check missing values --- 

# Count and percentage of missing values per column
missing_count = df.isna().sum()
missing_pct = missing_count / len(df)

missing_table = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)

print("\n=== Missing values summary ===")
print(missing_table)

# For this US dataset, there exists 0 missing values everywhere.
# Then we do not need to drop anything for missingness.

# --- 4. Select numerical feature columns and winsorize outliers --- 

# Financial feature columns are X1, X2, ..., X18
feature_cols = [f"X{i}" for i in range(1, 19)]
print("\n=== Feature columns ===")
print(feature_cols)

def winsorize_series(s, lower_quantile=0.01, upper_quantile=0.99):
    """
    Winsorize a numeric pandas Series:
    - Values below the lower_quantile are set to the lower_quantile value
    - Values above the upper_quantile are set to the upper_quantile value
    """
    lower = s.quantile(lower_quantile)
    upper = s.quantile(upper_quantile)
    return s.clip(lower=lower, upper=upper)

print("\n=== Winsorization (1st and 99th percentiles) ===")

for col in feature_cols:
    before_min, before_max = df[col].min(), df[col].max()
    df[col] = winsorize_series(df[col], 0.01, 0.99)
    after_min, after_max = df[col].min(), df[col].max()
    print(f"{col}: min {before_min:.2f} -> {after_min:.2f}, "
          f"max {before_max:.2f} -> {after_max:.2f}")

# --- 5. Simple checks on industry variables ---

print("\n=== Division value counts ===")
print(df["Division"].value_counts())

print("\n=== MajorGroup descriptive statistics ===")
print(df["MajorGroup"].describe())

# --- 6. Save the cleaned dataset ---

clean_path = "american_bankruptcy_cleaned.csv"
df.to_csv(clean_path, index=False)

print(f"\nCleaned dataset saved to: {clean_path}")
