import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Step 1: Normalize — scale pixels from 0-255 to 0-1
# Neural networks learn much faster with small numbers
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# Step 2: Reshape — flatten 28x28 grid into a 784-length list
# (for a basic dense network, not CNN yet)
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

print("After preprocessing:")
print("x_train shape:", x_train.shape)  # (60000, 784)
print("x_test shape:", x_test.shape)    # (10000, 784)
print("Pixel range:", x_train.min(), "to", x_train.max())  # 0.0 to 1.0

# Save for use in next steps
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
print("Saved preprocessed data!")