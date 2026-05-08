import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load dataset — downloads automatically, no manual work needed
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Understand the shape
print("Training images:", x_train.shape)   # (60000, 28, 28)
print("Training labels:", y_train.shape)   # (60000,)
print("Pixel value range:", x_train.min(), "to", x_train.max())  # 0 to 255

# Visualize 10 sample digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()