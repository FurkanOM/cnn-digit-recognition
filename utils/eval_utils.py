"""Model evaluation and reporting helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.common import DatasetSplit, EvaluationResults


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[Any],
    normalize: bool = True,
    title: Optional[str] = None,
    cmap: Any = plt.cm.Blues,
) -> Any:
    """Plot a confusion matrix with optional row-wise normalization.

    Args:
        cm (np.ndarray): Confusion matrix values.
        labels (Sequence[Any]): Class labels shown on the axes.
        normalize (bool): Whether to normalize the matrix by row totals.
        title (Optional[str]): Custom plot title.
        cmap (Any): Matplotlib color map.

    Returns:
        Any: Matplotlib axes that contain the rendered matrix.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    if normalize:
        numerator = cm.astype("float")
        denominator = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    fig, ax = plt.subplots()
    image = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.margins(0.05)

    value_format = ".2f" if normalize else "d"
    threshold = cm.max() / 2.0
    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            ax.text(
                column_index,
                row_index,
                format(cm[row_index, column_index], value_format),
                ha="center",
                va="center",
                color="white" if cm[row_index, column_index] > threshold else "black",
            )

    fig.tight_layout()
    return ax


def handle_confusion_matrix(
    actuals: Sequence[Any],
    predictions: Sequence[Any],
    labels: Sequence[Any],
    title: str,
    save_path: Optional[str] = None,
    print_results: bool = True,
) -> None:
    """Generate and optionally print confusion-matrix diagnostics.

    Args:
        actuals (Sequence[Any]): Ground-truth labels.
        predictions (Sequence[Any]): Predicted labels.
        labels (Sequence[Any]): Label order used by sklearn reports.
        title (str): Figure title.
        save_path (Optional[str]): Optional file path for the plot image.
        print_results (bool): Whether to print text diagnostics.

    Returns:
        None: Diagnostics are printed or written to disk.
    """
    cm = confusion_matrix(actuals, predictions)
    plot_confusion_matrix(cm, labels, title=title)
    if save_path:
        plt.savefig(save_path)
    if not print_results:
        return

    print("Classification Report")
    print(classification_report(actuals, predictions, labels=labels))
    print("Confusion Matrix")
    print(cm)
    print("Accuracy Score")
    print(accuracy_score(actuals, predictions))


def evaluate(model: Any, datasets: Mapping[str, DatasetSplit], use_datasets: Sequence[str]) -> EvaluationResults:
    """Evaluate a trained model on each requested dataset.

    Args:
        model (Any): Keras-compatible model with an `evaluate` method.
        datasets (Mapping[str, DatasetSplit]): Dataset collection.
        use_datasets (Sequence[str]): Dataset names used for evaluation.

    Returns:
        EvaluationResults: Rounded loss and accuracy values for each dataset.
    """
    results: EvaluationResults = {}
    for dataset_name in use_datasets:
        dataset = datasets[dataset_name]
        loss, accuracy = model.evaluate(dataset["x_test"], dataset["y_test"])
        results[dataset_name] = {
            "dataset": dataset_name,
            "loss": round(float(loss), 5),
            "accuracy": round(float(accuracy), 5),
        }
    return results


def print_results(results: Mapping[str, Mapping[str, Any]], trained_with: str, version: str) -> None:
    """Print evaluation results for a trained model.

    Args:
        results (Mapping[str, Mapping[str, Any]]): Evaluation payload.
        trained_with (str): Dataset combination used for training.
        version (str): Model version name.

    Returns:
        None: Results are printed to standard output.
    """
    print("============================================================================")
    print("Trained with:", trained_with, "model version", version)
    print("============================================================================")
    for dataset_name, result in results.items():
        print(dataset_name, "Test loss:", result["loss"])
        print(dataset_name, "Test accuracy:", result["accuracy"])
        print("=========================================")
    print("============================================================================")
