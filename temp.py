import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (4 features)
y = iris.target.reshape(-1, 1)  # Labels (3 classes)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# Normalize the feature data (X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalize features to have zero mean and unit variance

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

# Create DataFrames for CSV output
X_train_df = pd.DataFrame(X_train, columns=iris.feature_names)
X_test_df = pd.DataFrame(X_test, columns=iris.feature_names)

# For y_train and y_test, convert them into DataFrames with class labels
y_train_df = pd.DataFrame(y_train, columns=[f'class_{i}' for i in range(y_train.shape[1])])
y_test_df = pd.DataFrame(y_test, columns=[f'class_{i}' for i in range(y_test.shape[1])])

# Save the data to CSV files without the header (no column names) or index
X_train_df.to_csv('X_train.csv', header=False, index=False)
X_test_df.to_csv('X_test.csv', header=False, index=False)
y_train_df.to_csv('y_train.csv', header=False, index=False)
y_test_df.to_csv('y_test.csv', header=False, index=False)

print("CSV files for X_train, X_test, y_train, and y_test created.")
