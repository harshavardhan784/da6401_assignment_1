"""
Error Analysis Script — Q2.8
Loads the best saved model, runs inference on the test set, and logs:

  1. wandb.plot.confusion_matrix  — standard confusion matrix
  2. Misclassified-samples grid   — creative per-class failure visualization:
       For each true class, the 5 most confidently mis-classified examples
       are shown with their predicted label and the model's confidence score.
  3. Per-class accuracy bar chart — shows which classes the model struggles with

Usage (from da6401_assignment_1/src/):
    python -m error_analysis
    python -m error_analysis --model_path ../models/best_model.npy \
        --wandb_project da6401-a1
    python -m error_analysis --dataset fashion_mnist \
        -nhl 4 -sz 128 128 128 128
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.activations import softmax


FASHION_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Q2.8 - Error Analysis / Confusion Matrix")

    parser.add_argument("--model_path",    type=str, default="../models/best_model.npy")
    parser.add_argument("--wandb_project", type=str, default="da6401-a1")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--run_name",      type=str, default="error_analysis")

    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    # Architecture — must match the saved model exactly
    parser.add_argument("-nhl", "--num_layers",    type=int,          default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int, nargs="+",
                        default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",    type=str,          default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i", "--weight_init",   type=str,          default="xavier")
    parser.add_argument("-o",   "--optimizer",     type=str,          default="rmsprop")
    parser.add_argument("-lr",  "--learning_rate", type=float,        default=1e-3)
    parser.add_argument("-wd",  "--weight_decay",  type=float,        default=0.0)
    parser.add_argument("-l",   "--loss",          type=str,          default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    return parser.parse_args()


def _per_class_accuracy_fig(y_true, y_pred, class_names):
    """Bar chart of per-class accuracy reveals which classes the model struggles with."""
    n_classes = len(class_names)
    per_class_acc = []
    for cls in range(n_classes):
        mask = np.array(y_true) == cls
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(float(np.mean(np.array(y_pred)[mask] == cls)))

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["tomato" if a < 0.90 else "steelblue" for a in per_class_acc]
    bars = ax.bar(range(n_classes), per_class_acc, color=colors, edgecolor="white")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class Test Accuracy (red = below 90 %)")
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(y=np.mean(per_class_acc), color="black",
               linestyle="--", linewidth=1, label="mean accuracy")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def _misclassified_grid_fig(X_test_raw, y_true_arr, y_pred_arr,
                             probs, class_names, n_per_class=5):
    """
    Creative failure visualization:
    A grid where each ROW = one true class, each COLUMN = one of the top-N
    most confidently mis-classified examples for that class.
    Subtitle shows: predicted label and softmax confidence.

    X_test_raw  : (N, 784) float, values in any range (will normalize for display)
    probs       : (N, 10)  softmax probabilities
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_per_class,
                             figsize=(n_per_class * 1.8, n_classes * 1.8))
    fig.suptitle(
        "Most Confidently Mis-classified Examples\n"
        "(row = true class, col = top failures, subtitle = predicted class @ confidence)",
        fontsize=10, y=1.01,
    )

    for cls in range(n_classes):
        # Indices that BELONG to this true class but were mis-classified
        wrong_mask = (y_true_arr == cls) & (y_pred_arr != cls)
        wrong_idxs = np.where(wrong_mask)[0]

        if len(wrong_idxs) == 0:
            for col in range(n_per_class):
                axes[cls, col].axis("off")
            axes[cls, 0].set_ylabel(class_names[cls], fontsize=7, rotation=0,
                                    labelpad=40, va="center")
            continue

        # Sort by confidence of the WRONG prediction (most confident failures first)
        conf_of_wrong = probs[wrong_idxs, y_pred_arr[wrong_idxs]]
        sorted_order  = np.argsort(conf_of_wrong)[::-1]     # descending
        top_idxs      = wrong_idxs[sorted_order[:n_per_class]]

        for col, idx in enumerate(top_idxs):
            ax = axes[cls, col]
            img = X_test_raw[idx].reshape(28, 28)
            # Normalize for display
            img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img_disp, cmap="gray")
            pred_cls  = y_pred_arr[idx]
            conf      = float(probs[idx, pred_cls])
            ax.set_title(f"→{class_names[pred_cls]}\n{conf:.0%}",
                         fontsize=6, color="red")
            ax.axis("off")

        # Pad remaining columns if fewer than n_per_class failures exist
        for col in range(len(top_idxs), n_per_class):
            axes[cls, col].axis("off")

        axes[cls, 0].set_ylabel(class_names[cls], fontsize=7, rotation=0,
                                labelpad=40, va="center")

    plt.tight_layout()
    return fig


def main():
    args = parse_arguments()
    args.input_dim  = 784
    args.output_dim = 10

    class_names = (FASHION_CLASS_NAMES if args.dataset == "fashion_mnist"
                   else [str(i) for i in range(10)])

    # Load data
    print(f"Loading dataset: {args.dataset}")
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    # Load model 
    print(f"Loading model from: {args.model_path}")
    model = NeuralNetwork(args)
    model.load(args.model_path)

    # Inference — forward() returns logits; apply softmax to get probabilities
    print("Running inference...")
    logits = model.forward(X_test)
    probs  = softmax(logits)       # FIX: was incorrectly: probs = logits
    y_pred_arr = np.argmax(probs, axis=1)
    y_true_arr = np.argmax(y_test,  axis=1)

    y_pred = y_pred_arr.tolist()
    y_true = y_true_arr.tolist()

    accuracy = float(np.mean(y_pred_arr == y_true_arr))
    print(f"Test accuracy: {accuracy:.4f}")

    # Build figures 
    print("Building visualizations...")
    fig_acc  = _per_class_accuracy_fig(y_true, y_pred, class_names)

    # For the misclassified grid we need the *unnormalized* pixel images.
    # data_loader standardizes X_test; load raw images separately for display.
    if args.dataset == "fashion_mnist":
        from keras.datasets import fashion_mnist as _loader
        (_, _), (X_raw, _) = _loader.load_data()
    else:
        from keras.datasets import mnist as _loader
        (_, _), (X_raw, _) = _loader.load_data()
    X_raw_flat = X_raw.reshape(len(X_raw), -1).astype(np.float64)

    fig_fail = _misclassified_grid_fig(
        X_raw_flat, y_true_arr, y_pred_arr, probs, class_names, n_per_class=5
    )

    # Log to W&B 
    print("Logging to W&B...")
    run = wandb.init(
        project=args.wandb_project,
        entity =args.wandb_entity,
        name   =args.run_name,
        config ={
            "dataset":     args.dataset,
            "model_path":  args.model_path,
            "num_layers":  args.num_layers,
            "hidden_size": args.hidden_size,
            "activation":  args.activation,
        },
    )

    run.log({
        "test_accuracy":          accuracy,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=y_true, preds=y_pred,
            class_names=class_names,
        ),
        "per_class_accuracy":     wandb.Image(fig_acc),
        "misclassified_examples": wandb.Image(fig_fail),
    })

    plt.close("all")
    run.finish()
    print("Done! Check W&B for confusion matrix and failure visualizations.")


if __name__ == "__main__":
    main()