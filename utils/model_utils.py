import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.utils import plot_model

from .constants import IMG_PATH, INFO_PATH

rcParams["figure.figsize"] = (12, 10)
colors = plt.rcParams["axes.prop_cycle"].by_key()[
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

    return model


def plot_log_loss(history: History, title_label: str, version: str = "v1") -> ():
    plt.semilogy(
        history.epoch,
        history.history["loss"],
        color=colors[0],
        label="Train " + title_label,
    )
    plt.semilogy(
        history.epoch,
        history.history["val_loss"],
        color=colors[4],
        label="Val " + title_label,
        linestyle="--",
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
        plt.plot(history.epoch, history.history[metric], color=colors[0], label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[0],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)

    plt.legend()
    plt.savefig(f"{IMG_PATH}/{version}/metrics_{version}")


def evaluate_model(model: Model, test_ds: Dataset):
    print("Evaluate model")

    score_test = model.evaluate(test_ds.batch(batch_size=64))

    for name, value in zip(model.metrics_names, score_test):
        print(name, ": ", value)
