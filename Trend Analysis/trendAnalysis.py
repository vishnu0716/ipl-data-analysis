import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("ipl_dataset.csv")

print("Initial Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# ============================================================
# 3. DATA CLEANING
# ============================================================

df['winner'] = df['winner'].fillna('Unknown')
df['city'] = df['city'].fillna(df['city'].mode()[0])
df['player_of_match'] = df['player_of_match'].fillna('Not Awarded')
df['method'] = df['method'].fillna('Normal')

num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df = df.drop_duplicates()

df = df[df['winner'] != 'XYZ Team']
df = df[df['city'] != 'UnknownCity']

df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df.dropna()

print("\nCleaned Shape:", df.shape)

# FINAL CHECK
print("\nAfter Cleaning Missing Values:\n", df.isnull().sum())
print("\nDuplicates After Cleaning:", df.duplicated().sum())

df.to_csv("ipl_cleaned_dataset.csv", index=False)

# ============================================================
# ================= TREND ANALYSIS ===========================
# ============================================================

df['season'].value_counts().sort_index().plot(marker='o')
plt.title("Matches Over Seasons")
plt.xticks(rotation=90)
plt.show()

avg_runs = df.groupby('season')['target_runs'].mean()
avg_runs.plot(marker='o')
plt.title("Average Runs Trend")
plt.xticks(rotation=90)
plt.show()