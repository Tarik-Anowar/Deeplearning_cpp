import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the training and testing data from CSV files
X_train = pd.read_csv('X_train.csv', header=None).values
X_test = pd.read_csv('X_test.csv', header=None).values
y_train = pd.read_csv('y_train.csv', header=None).values
y_test = pd.read_csv('y_test.csv', header=None).values

# Check the shape of the data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Define the neural network model using Keras
model = models.Sequential()

# Add the input layer with 4 features (the same as the Iris dataset)
model.add(layers.InputLayer(input_shape=(4,)))

# Add a hidden layer with 64 neurons and ReLU activation
model.add(layers.Dense(64, activation='relu'))

# Add the output layer with 3 neurons (for 3 classes) and softmax activation
model.add(layers.Dense(3, activation='softmax'))

# Compile the model using Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the training data (epochs = 50, batch_size = 32)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model using the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Print the accuracy of the model
print(f'Test Accuracy: {accuracy * 100:.2f}%')
