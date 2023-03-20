import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.metrics import Precision
from tensorflow.keras.utils import plot_model

from .constants import IMG_PATH, INFO_PATH

rcParams["figure.figsize"] = (12, 10)
COLORS = plt.rcParams["axes.prop_cycle"].by_key()[
    "color"
]  # 7 colors: blue, orange, green, red, purple, brown, and pink


def save_img_model(model: Model, version: str = "v1"):
    new_path = f"{IMG_PATH}/{version}"

    if not os.path.exists(IMG_PATH):
        os.mkdir(IMG_PATH)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    plot_model(model, to_file=f"{new_path}/model_{version}.png", show_shapes=True)


def save_model_info(
    model: Model,
    early_callback: EarlyStopping,
    best_wigth: bool = True,
    version: str = "v1",
):
    if not os.path.exists(INFO_PATH):
        os.mkdir(INFO_PATH)

    model_path = f"{INFO_PATH}/{version}"

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if best_wigth:
        best_weights = early_callback.best_weights
        model.set_weights(best_weights)

    model.save(f"{model_path}/model_{version}.h5")
    model.save_weights(f"{model_path}/weights_{version}.h5")

    with open(f"{model_path}/best_epoch_{version}.txt", "w") as file:
        file.write(str(early_callback.best_epoch))

    return model


def plot_log_loss(history: History, title_label: str, version: str = "v1"):
    model_path = f"{INFO_PATH}/{version}"

    with open(f"{model_path}/best_epoch_{version}.txt", "r") as file:
        best_epoch = int(file.read())

    plt.semilogy(
        history.epoch,
        history.history["loss"],
        color=COLORS[0],
        label="Train " + title_label,
    )
    plt.semilogy(
        history.epoch,
        history.history["val_loss"],
        color=COLORS[4],
        label="Val " + title_label,
        linestyle="--",
    )
    plt.semilogy(
        history.epoch,
        history.history["val_auc"],
        color=COLORS[2],
        label="Val auc " + title_label,
        linestyle="--",
    )
    plt.axvline(
        x=history.epoch[best_epoch], color=COLORS[5], label=f"Best epoch {best_epoch}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.savefig(f"{IMG_PATH}/{version}/log_loss_{version}")


def plot_metrics(history: History, version: str = "v1"):
    metrics = ["loss", "precision", "recall", "auc", "tp", "sensitivity"]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(3, 2, n + 1)  # adjust according to metrics
        plt.plot(history.epoch, history.history[metric], color=COLORS[0], label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=COLORS[0],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)

    plt.legend()
    plt.savefig(f"{IMG_PATH}/{version}/metrics_{version}")


def evaluate_model(model: Model, test_ds: Dataset, version: str, batch_size: int = 10):
    score_test = model.evaluate(test_ds.batch(batch_size=batch_size))

    evaluation_results = {
        name: value for name, value in zip(model.metrics_names, score_test)
    }

    file_path = f"{INFO_PATH}/{version}/evaluation_{version}.json"

    with open(file_path, "w") as f:
        json.dump(evaluation_results, f)


def generate_predictions(model: Model, x_train: np.array, x_test: np.array):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    return y_train_pred, y_test_pred


def plot_confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, version: str, y_type: str, p: int = 0.5
):
    labels_str = ["No Cancer", "Cancer"]

    cm = confusion_matrix(labels, predictions > p)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels_str, yticklabels=labels_str)
    plt.title("Confusion matrix @{:.2f}".format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    plt.savefig(
        f"{IMG_PATH}/{version}/{y_type}_confusion_matrix_{version}",
        bbox_inches="tight",
        dpi=300,
    )

    tn, fp, fn, tp = cm.ravel()
    results = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_lesions": int(np.sum(cm[1])),
    }

    results_path = (
        f"{INFO_PATH}/{version}/{y_type}_classification_metrics_{version}.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f)


def save_precision_results(
    labels: np.ndarray, predictions: np.ndarray, version: str, y_type: str
):
    precision = Precision()
    precision.update_state(labels, predictions)
    precision_result = precision.result().numpy()

    results = {"precision": float(precision_result)}

    precision_path = f"{INFO_PATH}/{version}/{y_type}_precision_metric_{version}.json"

    with open(precision_path, "w") as f:
        json.dump(results, f)


def plot_roc_curves(datasets: List[dict], version: str):
    plt.figure(figsize=(8, 8))
    lw = 2

    for i, data in enumerate(datasets):
        name = data["name"]
        labels = data["labels"]
        predictions = data["predictions"]

        fp, tp, _ = roc_curve(labels, predictions)
        auc = roc_auc_score(labels, predictions)
        plt.plot(
            100 * fp,
            100 * tp,
            color=COLORS[i % len(COLORS)],
            lw=lw,
            label=f"{name} (AUC = {auc:.3f})",
        )

    plt.plot([0, 100], [0, 100], color="gray", lw=lw, linestyle="--")
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")

    plt.savefig(f"{IMG_PATH}/{version}/roc_curve_{version}")
