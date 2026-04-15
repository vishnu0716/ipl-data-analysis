import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro

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
# ================= NORMALITY TEST ===========================
# ============================================================

stat, p = shapiro(df['target_runs'])

print("\nNormality Test P-value:", p)

if p > 0.05:
    print("Normal Distribution")
else:
    print("Not Normal Distribution")

sns.histplot(df['target_runs'], kde=True)
plt.title("Distribution")
plt.show()