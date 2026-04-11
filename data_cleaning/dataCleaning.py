import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CODE CELL 2: LOAD DATA
# ============================================================

df = pd.read_csv("agriMarket_prices.csv")
print(df.head())


# ============================================================
# CODE CELL 3: BASIC CHECKS
# ============================================================

print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nMissing BEFORE:\n", df.isnull().sum())
print("\nInfo:\n")
df.info()


# ============================================================
# CODE CELL 4: DATA CLEANING (FIXED)
# ============================================================

# Convert types
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')

df['Min_Price'] = pd.to_numeric(df['Min_Price'], errors='coerce')
df['Max_Price'] = pd.to_numeric(df['Max_Price'], errors='coerce')
df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Fill numeric columns (by commodity group to avoid wrong propagation)
df[['Min_Price','Max_Price','Modal_Price']] = (
    df.groupby('Commodity')[['Min_Price','Max_Price','Modal_Price']]
    .transform(lambda x: x.ffill().bfill())
)

# Fill date column
df['Arrival_Date'] = df['Arrival_Date'].ffill().bfill()

# NOTE: Not using df.fillna(0) intentionally

print("\nMissing AFTER:\n", df.isnull().sum())


# ============================================================
# CODE CELL 5: FEATURE EXTRACTION
# ============================================================

df['Year'] = df['Arrival_Date'].dt.year
df['Month'] = df['Arrival_Date'].dt.month

print("\nDescribe:\n", df.describe())