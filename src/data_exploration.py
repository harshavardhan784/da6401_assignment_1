"""
Data Exploration Script — Q2.1
Logs to W&B:
  • wandb.Table with 5 sample images from each of the 10 classes
  • Class-distribution bar chart
  • Per-class mean-image grid  (reveals which classes look alike)
  • Inter-class pixel-similarity heatmap (cosine similarity of mean images)

Usage (from da6401_assignment_1/src/):
    python -m data_exploration
    python -m data_exploration --wandb_project da6401-a1
    python -m data_exploration --wandb_project da6401-a1 --dataset fashion_mnist
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from keras.datasets import mnist, fashion_mnist


# ─────────────────────────────────────────────────────────────────────────────
FASHION_CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
MNIST_CLASS_NAMES = [str(i) for i in range(10)]


# ─────────────────────────────────────────────────────────────────────────────
def _class_distribution_fig(counts, class_names, ds_name):
    """Return a bar-chart Figure for the class distribution."""
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(10), counts, color="steelblue", edgecolor="white")
    ax.set_xticks(range(10))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"{ds_name} — Class Distribution (train split)")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            str(count),
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    return fig


def _mean_image_grid_fig(X, y, class_names, ds_name):
    """
    Return a grid figure showing the mean image for every class.
    Visually similar mean images indicate classes the model may confuse.
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"{ds_name} — Per-Class Mean Image", fontsize=13)
    for cls, ax in enumerate(axes.flat):
        mask = (y == cls)
        mean_img = X[mask].mean(axis=0).reshape(28, 28)
        ax.imshow(mean_img, cmap="gray")
        ax.set_title(class_names[cls], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig


def _similarity_heatmap_fig(X, y, class_names, ds_name):
    """
    Return a cosine-similarity heatmap between every pair of class mean images.
    High values → visually similar classes → harder for the model to distinguish.
    """
    means = np.array([
        X[y == cls].mean(axis=0).flatten()
        for cls in range(10)
    ], dtype=np.float64)

    # Cosine similarity: dot(a,b) / (||a|| * ||b||)
    norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-12
    normed = means / norms
    sim = normed @ normed.T

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title(
        f"{ds_name} — Inter-class Pixel Similarity\n"
        "(cosine similarity of mean images — high = visually similar)",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Annotate cells
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{sim[i,j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if sim[i, j] < 0.8 else "white")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def explore_dataset(ds_name: str, loader, class_names: list, project: str, entity: str):
    """Run exploration for one dataset and push all artefacts to W&B."""
    print(f"\nExploring: {ds_name}")
    (X, y), _ = loader.load_data()          # raw uint8 images
    X_flat = X.reshape(len(X), -1).astype(np.float64)   # (N, 784) for similarity

    run = wandb.init(
        project=project,
        entity=entity,
        name=f"data_exploration_{ds_name}",
        config={"dataset": ds_name},
    )

    # ── 1. Sample images table (5 images per class) ──────────────────────
    table = wandb.Table(columns=["class_id", "class_name", "image"])
    for cls in range(10):
        idxs = np.where(y == cls)[0][:5]
        for i in idxs:
            table.add_data(cls, class_names[cls], wandb.Image(X[i]))

    # ── 2. Class distribution bar chart ──────────────────────────────────
    counts = np.bincount(y)
    fig_dist = _class_distribution_fig(counts, class_names, ds_name)

    # ── 3. Mean-image grid (shows which classes share visual structure) ───
    fig_means = _mean_image_grid_fig(X_flat, y, class_names, ds_name)

    # ── 4. Inter-class cosine similarity heatmap ─────────────────────────
    fig_sim = _similarity_heatmap_fig(X_flat, y, class_names, ds_name)

    # ── Push to W&B ───────────────────────────────────────────────────────
    run.log({
        "samples":                  table,
        "class_distribution":       wandb.Image(fig_dist),
        "mean_images_per_class":    wandb.Image(fig_means),
        "inter_class_similarity":   wandb.Image(fig_sim),
    })

    plt.close("all")
    run.finish()
    print(f"  Done — W&B project '{project}' › run: data_exploration_{ds_name}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Q2.1 — Data Exploration")
    parser.add_argument("--wandb_project", type=str, default="da6401-a1")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default="both",
        choices=["mnist", "fashion_mnist", "both"],
        help="Dataset(s) to explore (default: both)",
    )
    args = parser.parse_args()

    datasets = {
        "mnist":         (mnist,         MNIST_CLASS_NAMES),
        "fashion_mnist": (fashion_mnist, FASHION_CLASS_NAMES),
    }
    to_run = list(datasets.keys()) if args.dataset == "both" else [args.dataset]

    for ds_name in to_run:
        loader, class_names = datasets[ds_name]
        explore_dataset(ds_name, loader, class_names,
                        project=args.wandb_project,
                        entity=args.wandb_entity)

    print("\nAll done!")


if __name__ == "__main__":
    main()