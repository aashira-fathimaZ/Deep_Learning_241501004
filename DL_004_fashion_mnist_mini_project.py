"""
DEEP LEARNING MINI PROJECT
Fashion Item Image Classification: Baseline Neural Network vs CNN
Dataset : Fashion-MNIST (70,000 grayscale 28x28 images, 10 clothing classes)
Library : TensorFlow / Keras

====================================================================
1. AIM OF THE PROJECT
====================================================================
This project classifies grayscale images of clothing items (T-shirt, trouser,
sneaker, bag, etc.) into 10 categories using deep learning on the Fashion-MNIST
dataset. A simple fully connected neural network is built as the baseline and
compared with an improved Convolutional Neural Network (CNN) using accuracy,
precision, recall, F1-score and confusion matrices.

====================================================================
2. PROCEDURE
====================================================================
1. Import the required libraries.
2. Load the Fashion-MNIST dataset (60,000 training and 10,000 test images).
3. Preprocess the data: normalize pixels to the range 0-1 and reshape images.
4. Split the training data into training (80%) and validation (20%) sets.
5. Build the baseline model: a simple fully connected neural network (MLP).
6. Build the improved Deep Learning model: a Convolutional Neural Network (CNN)
   with Conv2D, MaxPooling, Dropout and Dense layers.
7. Train both models on the training set, monitoring the validation set.
8. Evaluate both models on the unseen test set using accuracy, precision,
   recall, F1-score and confusion matrix.
9. Compare the results of both models and draw conclusions.

====================================================================
3. CODE
====================================================================
"""

# ---------------------------------------------------------------
# Step 1: Import required libraries
# ---------------------------------------------------------------
import os
import gzip
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # lets the script run without a display
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Fix random seeds so results are reproducible
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
EPOCHS = 10          # increase (e.g. 15-20) for slightly better accuracy
BATCH_SIZE = 128


# ---------------------------------------------------------------
# Step 2: Load the dataset
# ---------------------------------------------------------------
def load_fashion_mnist():
    """Load Fashion-MNIST using Keras. If the Keras download fails
    (e.g. blocked network), fall back to downloading the same files from GitHub."""
    try:
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
        return x_tr, y_tr, x_te, y_te
    except Exception as err:
        print("Keras download failed (", type(err).__name__, "). Using GitHub fallback...")

    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    os.makedirs("fashion_data", exist_ok=True)
    for f in files:
        path = os.path.join("fashion_data", f)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + f, path)

    def read_images(path):                 # IDX format: 16-byte header, then pixels
        with gzip.open(path, "rb") as fh:
            return np.frombuffer(fh.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    def read_labels(path):                 # IDX format: 8-byte header, then labels
        with gzip.open(path, "rb") as fh:
            return np.frombuffer(fh.read(), np.uint8, offset=8)

    return (read_images("fashion_data/" + files[0]), read_labels("fashion_data/" + files[1]),
            read_images("fashion_data/" + files[2]), read_labels("fashion_data/" + files[3]))


x_train_full, y_train_full, x_test, y_test = load_fashion_mnist()
print("Training images:", x_train_full.shape, " Test images:", x_test.shape)

# Show a few sample images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_full[i], cmap="gray")
    plt.title(CLASS_NAMES[y_train_full[i]], fontsize=8)
    plt.axis("off")
plt.suptitle("Sample images from Fashion-MNIST")
plt.tight_layout()
plt.savefig("sample_images.png")
plt.close()


# ---------------------------------------------------------------
# Step 3: Preprocess the data
# ---------------------------------------------------------------
# Pixel values are 0-255. Scaling to 0-1 helps the network train faster.
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNNs expect a channel dimension: (28, 28) -> (28, 28, 1)
x_train_full = x_train_full[..., np.newaxis]
x_test = x_test[..., np.newaxis]


# ---------------------------------------------------------------
# Step 4: Split the dataset (80% train / 20% validation of the 60,000 images)
# ---------------------------------------------------------------
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full)
print("Train:", x_train.shape, " Validation:", x_val.shape, " Test:", x_test.shape)


# ---------------------------------------------------------------
# Step 5: Build the BASELINE model (simple fully connected network)
# ---------------------------------------------------------------
def build_baseline():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),                         # image -> 784 numbers
        layers.Dense(128, activation="relu"),     # one hidden layer
        layers.Dense(10, activation="softmax"),   # 10 class probabilities
    ], name="baseline_mlp")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------
# Step 6: Build the IMPROVED model (Convolutional Neural Network)
# ---------------------------------------------------------------
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),  # learns edges/textures
        layers.MaxPooling2D((2, 2)),                                   # shrink image by half
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),  # learns shapes
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),                                           # reduces overfitting
        layers.Dense(10, activation="softmax"),
    ], name="improved_cnn")
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


baseline_model = build_baseline()
cnn_model = build_cnn()
baseline_model.summary()
cnn_model.summary()


# ---------------------------------------------------------------
# Step 7: Train both models
# ---------------------------------------------------------------
print("\nTraining baseline model...")
hist_base = baseline_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                               epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

print("\nTraining CNN model...")
hist_cnn = cnn_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# Plot training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for hist, name in [(hist_base, "Baseline"), (hist_cnn, "CNN")]:
    ax[0].plot(hist.history["val_accuracy"], label=name + " validation")
    ax[0].plot(hist.history["accuracy"], "--", label=name + " training")
    ax[1].plot(hist.history["val_loss"], label=name + " validation")
    ax[1].plot(hist.history["loss"], "--", label=name + " training")
ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
for a in ax:
    a.set_xlabel("Epoch"); a.legend()
plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()


# ---------------------------------------------------------------
# Step 8: Evaluate both models on the unseen test set
# ---------------------------------------------------------------
def evaluate(model, name):
    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    results = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        # 'macro' = average of the 10 per-class scores, each class counts equally
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall":    recall_score(y_test, y_pred, average="macro"),
        "F1-score":  f1_score(y_test, y_pred, average="macro"),
    }
    print("\n===== " + name + " : Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    return results, confusion_matrix(y_test, y_pred), y_pred


base_results, base_cm, base_pred = evaluate(baseline_model, "Baseline MLP")
cnn_results, cnn_cm, cnn_pred = evaluate(cnn_model, "Improved CNN")

# Confusion matrices side by side
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
for a, cm, title in [(ax[0], base_cm, "Baseline MLP"), (ax[1], cnn_cm, "Improved CNN")]:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=a,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    a.set_title("Confusion Matrix - " + title)
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.close()


# ---------------------------------------------------------------
# Step 9: Compare the results
# ---------------------------------------------------------------
print("\n================ MODEL COMPARISON (Test Set) ================")
print("{:<12}{:>14}{:>14}{:>14}".format("Metric", "Baseline MLP", "Improved CNN", "Difference"))
for metric in base_results:
    b, c = base_results[metric], cnn_results[metric]
    print("{:<12}{:>14.4f}{:>14.4f}{:>+14.4f}".format(metric, b, c, c - b))
print("{:<12}{:>14,}{:>14,}".format("Parameters",
                                    baseline_model.count_params(), cnn_model.count_params()))

# Per-class accuracy: which clothing types are hardest?
print("\nPer-class accuracy (Baseline -> CNN):")
for i, name in enumerate(CLASS_NAMES):
    b = base_cm[i, i] / base_cm[i].sum()
    c = cnn_cm[i, i] / cnn_cm[i].sum()
    print("  {:<12} {:.3f} -> {:.3f}".format(name, b, c))

# Most confused pair of classes for the CNN (ignoring correct predictions)
cm_off = cnn_cm.copy()
np.fill_diagonal(cm_off, 0)
a, p = np.unravel_index(np.argmax(cm_off), cm_off.shape)
print("\nCNN's most common mistake: '{}' predicted as '{}' ({} times)".format(
    CLASS_NAMES[a], CLASS_NAMES[p], cm_off[a, p]))

# Bar chart of metrics
labels = list(base_results.keys())
xpos = np.arange(len(labels))
plt.figure(figsize=(8, 4))
plt.bar(xpos - 0.2, [base_results[m] for m in labels], 0.4, label="Baseline MLP")
plt.bar(xpos + 0.2, [cnn_results[m] for m in labels], 0.4, label="Improved CNN")
plt.xticks(xpos, labels)
plt.ylim(0.8, 1.0)
plt.ylabel("Score")
plt.title("Baseline vs CNN (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

print("\nCONCLUSION:")
print("1. The CNN scores higher than the baseline MLP on every metric above, because")
print("   convolution layers look at small local patches and learn edges, textures and")
print("   shapes, while the MLP flattens the image and loses the 2-D structure.")
print("2. Both models find upper-body items (Shirt, T-shirt, Pullover, Coat) hardest")
print("   to separate, since they look alike; footwear, Trouser and Bag are easy.")
print("3. Check training_curves.png: a small gap between training and validation")
print("   accuracy means little overfitting. Dropout in the CNN is intended to help")
print("   with this. Training for more epochs or adding data augmentation could")
print("   improve the CNN further.")
print("Saved plots: sample_images.png, training_curves.png, confusion_matrices.png, model_comparison.png")
