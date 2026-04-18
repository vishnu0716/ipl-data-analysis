import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.weightstats import ztest

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
# ================= HYPOTHESIS TEST ==========================
# ============================================================

team1 = df[df['winner'] == 'Chennai Super Kings']['target_runs']
team2 = df[df['winner'] == 'Mumbai Indians']['target_runs']

t_stat, t_p = ttest_ind(team1, team2)
print("\nT-Test P-value:", t_p)

z_stat, z_p = ztest(df['target_runs'], value=150)
print("Z-Test P-value:", z_p)

table = pd.crosstab(df['toss_winner'], df['winner'])
chi_stat, chi_p, _, _ = chi2_contingency(table)
print("Chi-Square P-value:", chi_p)