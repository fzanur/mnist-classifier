import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load data
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test  = np.load('x_test.npy')
y_test  = np.load('y_test.npy')

# Rebuild model (same as Day 3)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=10,               # 10 passes through the full dataset
    batch_size=32,           # learn from 32 images at a time
    validation_split=0.1,    # use 10% of training data to monitor overfitting
    verbose=1
)

# Evaluate on test set (data model has NEVER seen)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")  # aim for 0.97+

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Save the trained model
model.save('mnist_model.h5')
print("Model saved!")