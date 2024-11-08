import pickle
import numpy as np

# Load the pickle file
with open('mnist_subset_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']

# Define the split ratio (e.g., 80% training, 20% testing)
split_ratio = 0.8
split_index = int(len(X_train) * split_ratio)

# Split the data
X_train_split, X_test_split = X_train[:split_index], X_train[split_index:]
y_train_split, y_test_split = y_train[:split_index], y_train[split_index:]

# Save the split data to CSV files
np.savetxt('X_train.csv', X_train_split, delimiter=',')
np.savetxt('y_train.csv', y_train_split, delimiter=',')
np.savetxt('X_test.csv', X_test_split, delimiter=',')
np.savetxt('y_test.csv', y_test_split, delimiter=',')
