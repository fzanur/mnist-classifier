# 🔢 Handwritten Digit Classifier — MNIST

A neural network that recognizes handwritten digits (0–9) with **97.76% accuracy**, trained on the MNIST dataset using TensorFlow and Keras.

---

## 📌 Project Overview

This project builds, trains, and evaluates a deep learning model to classify handwritten digits from images. The model is trained on 60,000 images and tested on 10,000 unseen images, achieving 97.76% test accuracy.

---

## 🧠 Model Architecture

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| 1 | Dense (ReLU) | (None, 128) | 100,480 |
| 2 | Dropout (0.2) | (None, 128) | 0 |
| 3 | Dense (ReLU) | (None, 64) | 8,256 |
| 4 | Dense (Softmax) | (None, 10) | 650 |

**Total Parameters:** 109,386  
**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Epochs:** 10 | **Batch Size:** 32

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **97.76%** |
| Test Loss | ~0.08 |

### Training History
<img width="1200" height="400" alt="training_history" src="https://github.com/user-attachments/assets/e0ffa498-e23b-436f-b173-a89d4c44b407" />

### Confusion Matrix
The confusion matrix below shows which digits the model confuses most often. The most common misclassifications occurred between digits **3, 7, and 9** — which share visually similar curved and diagonal strokes in handwritten form.

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/aca8525c-bfec-4edb-83ad-2bb3c6a54db4" />

### Wrong Predictions
Examples of images the model misclassified:

<img width="1200" height="500" alt="wrong_predictions" src="https://github.com/user-attachments/assets/a2b0cf46-9696-4d65-9d99-1bff60eaecce" />

---

## 📁 Project Structure

```
mnist-classifier/
├── explore.py          # Load and visualize the dataset
├── preprocess.py       # Normalize and reshape data
├── model.py            # Define the neural network architecture
├── train.py            # Train the model and plot results
├── evaluate.py         # Evaluate with confusion matrix and error analysis
├── training_history.png
├── confusion_matrix.png
├── wrong_predictions.png
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/fzanur/mnist-classifier.git
cd mnist-classifier
```

**2. Install dependencies**
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

**3. Run in order**
```bash
python explore.py       # Visualize sample digits
python preprocess.py    # Preprocess and save data
python train.py         # Train the model (~5 mins)
python evaluate.py      # Evaluate and generate plots
```

> The MNIST dataset downloads automatically on first run — no manual download needed.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — model building and training
- **NumPy** — data handling
- **Matplotlib** — visualizations
- **Scikit-learn** — confusion matrix and classification report

---

## 📚 Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels:
- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Pixel values:** 0–255 (normalized to 0–1 during preprocessing)

---

## 👩‍💻 Author

**Fiza Noor**  
BS Artificial Intelligence — University of Engineering and Technology (UET), Lahore  
[LinkedIn](https://www.linkedin.com/in/fiza-noor-70b797323/) | [GitHub](https://github.com/fzanur)
