from __future__ import annotations
import numpy as np
import matplotlib
# Use non-interactive Agg backend: no display required on HPC (no DISPLAY env var)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_color_palette(n_classes: int) -> list:
    """Generate a color palette of length n_classes using tab20 (extended if needed)."""
    base_cmap = plt.cm.get_cmap("tab20")
    # tab20 has 20 distinct colours; use it for the first 20 classes
    colors = [base_cmap(i) for i in range(min(20, n_classes))]
    if n_classes > 20:
        # Extend with tab20b colours for any additional classes
        extra_cmap = plt.cm.get_cmap("tab20b")
        colors.extend([extra_cmap(i) for i in range(n_classes - 20)])
    return colors[:n_classes]


def plot_classification_map(
    data: np.ndarray,
    title: str,
    cmap,
    class_names: list,
    save_path: str,
    figsize: tuple = (12, 10),
):
    """Save a classification map PNG with a legend (no colorbar)."""
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.linewidth": 1.5})
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    # interpolation='nearest' preserves discrete class boundaries (no smoothing)
    ax.imshow(data, cmap=cmap, interpolation="nearest")

    if class_names:
        # Build a legend from the class values that actually appear in the data
        unique_classes = sorted(np.unique(data))
        # Exclude background (value 0) from the legend
        unique_classes = [c for c in unique_classes if c > 0]
        max_cls = max(unique_classes) if unique_classes else 1
        legend_patches = []
        for cls in unique_classes:
            # Map class integer to the same normalised color the cmap uses for imshow
            color = cmap(cls / max_cls)
            label = class_names[cls - 1] if cls - 1 < len(class_names) else f"Class {cls}"
            legend_patches.append(mpatches.Patch(color=color, label=label))
        ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),   # place legend outside the axes on the right
            loc="upper left",
            fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            title="Classes",
            title_fontsize=15,
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # release memory — essential when running many plots in a loop


def plot_per_class_accuracy(
    class_accuracies: dict,
    class_names: list,
    overall_accuracy: float,
    model_name: str,
    cmap,
    save_path: str,
):
    """Save a horizontal bar chart of per-class accuracy."""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    class_labels = [
        class_names[cls - 1] if cls - 1 < len(class_names) else f"Class {cls}"
        for cls in classes
    ]
    # Sort ascending so the best class appears at the top of the horizontal bar chart
    sorted_indices = np.argsort(accuracies)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    sorted_labels = [class_labels[i] for i in sorted_indices]

    max_cls = max(classes) if classes else 1
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
    ax.barh(
        range(len(sorted_classes)),
        sorted_accuracies,
        # Colour each bar with the same class colour used in the classification map
        color=[cmap(cls / max_cls) for cls in sorted_classes],
    )
    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels(sorted_labels, fontsize=12)
    ax.set_xlabel("Accuracy (%)", fontsize=14)
    ax.set_title(f"{model_name} - Per-Class Accuracy", fontsize=16)
    # Dashed vertical line marks the overall (pixel) test accuracy for reference
    ax.axvline(overall_accuracy, color="red", linestyle="--",
               label=f"Overall Accuracy: {overall_accuracy:.1f}%")
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    valid_classes: set,
    class_names: list,
    title: str,
    save_path: str,
):
    """Save a confusion matrix heatmap PNG."""
    # Sort class labels for a consistent row/column order
    labels_sorted = sorted(valid_classes)
    display_labels = [
        class_names[c - 1] if c - 1 < len(class_names) else f"Class {c}"
        for c in labels_sorted
    ]
    # Compute the confusion matrix using only the valid class labels
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=display_labels)
    # xticks_rotation=45 prevents label overlap for long class names
    disp.plot(ax=ax, xticks_rotation=45)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
