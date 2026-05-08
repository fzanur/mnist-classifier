import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load
model = tf.keras.models.load_model('mnist_model.h5')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Get predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # pick digit with highest probability

# --- Confusion Matrix ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — Which digits get confused?')
plt.savefig('confusion_matrix.png')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# --- Show some wrong predictions ---
wrong_idx = np.where(y_pred != y_test)[0]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = wrong_idx[i]
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}", color='red')
    ax.axis('off')
plt.suptitle("Examples the model got WRONG")
plt.tight_layout()
plt.savefig('wrong_predictions.png')
plt.show()