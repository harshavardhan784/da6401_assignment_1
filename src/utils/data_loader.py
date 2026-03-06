"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
Normalization: X / 255.0  (scale pixels to [0, 1])
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
from utils import set_seed

# One-hot encoding
def one_hot_encode(y, num_classes=10):
    y = y.astype(int)
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


# Flatten images (28x28 -> 784)
def flatten_images(X):
    return X.reshape(X.shape[0], -1).astype(np.float64)


# Data loader
def load_data(dataset="mnist", val_split=0.1, seed=42):
    """
    Returns:
        X_train, y_train
        X_val, y_val
        X_test, y_test
    All X arrays are normalized to [0, 1] via /255.
    """

    set_seed(seed)

    # Load dataset
    if dataset.lower() == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    elif dataset.lower() in ["fashion_mnist", "fashion-mnist"]:
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    # Flatten images
    X_train_full = flatten_images(X_train_full)
    X_test = flatten_images(X_test)

    # One-hot labels
    y_train_full_oh = one_hot_encode(y_train_full)
    y_test_oh = one_hot_encode(y_test)

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full_oh,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=y_train_full
    )

    # Normalize pixels to [0, 1]
    X_train = X_train / 255.0
    X_val   = X_val   / 255.0
    X_test  = X_test  / 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test_oh


# Test run
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("mnist")

    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    print("\nTrain min/max:", X_train.min(), X_train.max())