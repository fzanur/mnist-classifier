import tensorflow as tf
import numpy as np

# Load preprocessed data
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    # 128 neurons, relu kills negative values — helps model learn complex patterns

    tf.keras.layers.Dropout(0.2),
    # Randomly turns off 20% of neurons during training — prevents memorization

    tf.keras.layers.Dense(64, activation='relu'),
    # Second hidden layer — extracts deeper patterns

    tf.keras.layers.Dense(10, activation='softmax')
    # 10 output neurons = 10 digits (0-9)
    # softmax converts outputs to probabilities that sum to 1
])

# Compile — tell the model HOW to learn
model.compile(
    optimizer='adam',           # adam adjusts learning rate automatically
    loss='sparse_categorical_crossentropy',  # standard for multi-class classification
    metrics=['accuracy']
)

model.summary()  # prints architecture — read this carefully!