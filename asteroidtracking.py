#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns


path = '/home/root1/'
os.chdir(path)
sys.path.append(path)

# Step 1: Load Data

df = pd.read_csv("nasa.csv")

print(df.head())
print(df.info())

# Step 2: Select Features

features = [
    'Semi Major Axis', 'Eccentricity', 'Inclination',
    'Perihelion Arg', 'Asc Node Longitude', 'Mean Anomaly',
    'Perihelion Distance', 'Absolute Magnitude',
    'Est Dia in KM(min)', 'Relative Velocity km per sec'
]

target = 'Miss Dist.(Astronomical)'

# Drop missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Step 3: Log Transform (important for skewed data)

y_log = np.log1p(y)

# Step 4: Train-Test Split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Step 5: Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 6: Train Model

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Step 7: Predictions

y_pred_log = model.predict(X_test_scaled)

# Convert back from log scale

y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)


# Step 8: Evaluation


mae = mean_absolute_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n Model Performance:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")


# Step 9: Feature Importance

plt.figure(figsize=(10,5))
sns.barplot(x=model.feature_importances_, y=features)
plt.title("Feature Importance")
plt.show()

# Step 10: Actual vs Predicted

plt.figure(figsize=(10,6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
plt.xlabel("Actual Distance")
plt.ylabel("Predicted Distance")
plt.title("Actual vs Predicted")
plt.show()


# Step 11: Residual Plot

residuals = y_test_actual - y_pred_actual

plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residual Distribution")
plt.show()


# Step 12: Correlation Heatmap

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

from sklearn.linear_model import LinearRegression

# Train Linear Regression

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict (log scale)

y_pred_lr_log = lr_model.predict(X_test_scaled)

# Convert back

y_pred_lr = np.expm1(y_pred_lr_log)

# Evaluate
mae_lr = mean_absolute_error(y_test_actual, y_pred_lr)
mse_lr = mean_squared_error(y_test_actual, y_pred_lr)
r2_lr = r2_score(y_test_actual, y_pred_lr)

print("\n Linear Regression Performance:")
print(f"MAE: {mae_lr:.4f}")
print(f"MSE: {mse_lr:.4f}")
print(f"R² Score: {r2_lr:.4f}")
print(model.feature_importances_)