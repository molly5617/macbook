import tensorflow as tf
from tensorflow import keras


# Load the dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Print the shape of the data
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

# Normalize the data
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Create a simple model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# Evaluate the model
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
print("Test MSE:", test_mse_score)
print("Test MAE:", test_mae_score)