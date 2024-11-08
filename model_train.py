import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the pickle file
with open('mnist_subset_data.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X_train']
y = data['y_train']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with a loss function, optimizer, and metrics
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training set
model.fit(X_train, y_train, epochs=100)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save('mnist_trained_model.h5')
print("Model trained and saved to 'mnist_trained_model.h5'")
