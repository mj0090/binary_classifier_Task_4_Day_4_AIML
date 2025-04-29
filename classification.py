# Importing all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

# Loading the data
df = pd.read_csv("Breast_cancer_data.csv")
# Data Cleaning & Preprocessing

# Dropping unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])

# Encode target variable: M=1 (malignant), B=0 (benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Split features and target
X = df.drop(columns='diagnosis')
y = df['diagnosis']

# Train/Test Split and Standardize

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate Model

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("Confusion Matrix:\n", cm)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

# Plotting ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# Threshold Tuning

threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)

print(f"Precision (threshold={threshold}):", precision_score(y_test, y_pred_custom))
print(f"Recall (threshold={threshold}):", recall_score(y_test, y_pred_custom))

'''Sigmoid Function: Core of Logistic Regression
The logistic regression model calculates:
P(y=1∣x)=σ(z)= 1/1+e^−z, where z=w⋅x+b
The sigmoid function outputs a probability from 0 to 1.'''