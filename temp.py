import numpy as np
import pickle
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target.reshape(-1, 1)  # Labels

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# Select a subset (for example, 100 samples)
X_subset, _, y_subset, _ = train_test_split(X, y_onehot, train_size=100, stratify=y, random_state=42)

# Prepare the data for pickle
data = {
    'X': X_subset,
    'y': y_subset,
}

# Save the subset of data as a pickle file (binary format)
with open('iris_subset_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Subset of dataset (100 examples) prepared and saved to 'iris_subset_data.pkl'")
