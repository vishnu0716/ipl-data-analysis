# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 2. LOAD DATA (DIRECT)
# ============================================================

df = pd.read_csv("ipl_dataset.csv")

print("Data Loaded Successfully")
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Fill missing values
df['winner'] = df['winner'].fillna('Unknown')
df['city'] = df['city'].fillna(df['city'].mode()[0])
df['player_of_match'] = df['player_of_match'].fillna('Not Awarded')
df['method'] = df['method'].fillna('Normal')

# Fill numeric columns
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Remove duplicates
df = df.drop_duplicates()

# Convert date
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Drop remaining nulls
df = df.dropna()

print("\nData Cleaned Successfully")
print("Shape after cleaning:", df.shape)

# ============================================================
# 4. LINEAR REGRESSION
# ============================================================

# Features and target
X = df[['target_runs']]
y = df['result_margin']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nLinear Regression Results:")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))