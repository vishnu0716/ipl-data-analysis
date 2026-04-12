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
# VISUALIZATIONS
# ============================================================

team_wins = df['winner'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(x=team_wins.index, y=team_wins.values)
plt.xticks(rotation=90)
plt.title("Team Wins")
plt.show()

# Toss Decision
sns.countplot(x='toss_decision', data=df)
plt.title("Toss Decision")
plt.show()

# Win Margin Distribution
sns.histplot(df['result_margin'], kde=True)
plt.title("Win Margin Distribution")
plt.show()

# Scatter Plot
sns.scatterplot(x='target_runs', y='result_margin', data=df)
plt.title("Runs vs Margin")
plt.show()

# ============================================================
# PIE CHART (TOP TEAMS)
# ============================================================

top_teams = df['winner'].value_counts().head(6)

plt.figure(figsize=(6,6))
plt.pie(top_teams.values, labels=top_teams.index, autopct='%1.1f%%')
plt.title("Top Teams Win Share")
plt.show()
