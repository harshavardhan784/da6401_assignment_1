"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
Normalization: X / 255.0  (scale pixels to [0, 1])
"""

import numpy as np
from sklearn.model_selection import train_test_split

try:
    from utils import set_seed
except ImportError:
    import random
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)


def _load_keras_datasets():
    """Lazy import of keras datasets — supports both tensorflow.keras and standalone keras."""
    try:
        from keras.datasets import mnist, fashion_mnist
        return mnist, fashion_mnist
    except ImportError:
        pass
    raise ImportError(
        "Cannot import mnist/fashion_mnist. "
        "Please install tensorflow: pip install tensorflow"
    )


def one_hot_encode(y, num_classes=10):
    y = y.astype(int)
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


def flatten_images(X):
    return X.reshape(X.shape[0], -1).astype(np.float64)


def load_data(dataset="mnist", val_split=0.1, seed=42):
    """
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    All X arrays are normalized to [0, 1] via /255.
    """
    set_seed(seed)

    mnist_mod, fashion_mnist_mod = _load_keras_datasets()

    if dataset.lower() == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist_mod.load_data()
    elif dataset.lower() in ["fashion_mnist", "fashion-mnist"]:
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist_mod.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    X_train_full = flatten_images(X_train_full)
    X_test = flatten_images(X_test)

    y_train_full_oh = one_hot_encode(y_train_full)
    y_test_oh = one_hot_encode(y_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full_oh,
        test_size=val_split, random_state=seed,
        shuffle=True, stratify=y_train_full
    )

    X_train = X_train / 255.0
    X_val   = X_val   / 255.0
    X_test  = X_test  / 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test_oh


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data("mnist")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("Train min/max:", X_train.min(), X_train.max())