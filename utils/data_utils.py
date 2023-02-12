import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from .constants import IMG_PATH, INFO_PATH


def count_images_and_keep(filepaths: list):
    dict_files = {}

    for path in filepaths:
        images_files = []

        for dir_name, _, file_list in os.walk(path):
            for filename in file_list:
                if ".dcm" in filename.lower():
                    images_files.append(os.path.join(dir_name, filename))
        dict_files[path] = images_files

    return dict_files


def train_test_as_tensor(
    x: np.array, y: np.array, test_size: float = 0.2, validation_size: float = 0.25
):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=0
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=validation_size, random_state=0
    )

    train_ds = Dataset.from_tensor_slices((x_train, y_train))
    test_ds = Dataset.from_tensor_slices((x_test, y_test))
    validation_ds = Dataset.from_tensor_slices((x_val, y_val))

    return train_ds, test_ds, validation_ds


def save_img_model(model: Model):
    new_path = f"{IMG_PATH}/architectures"

    if not os.path.exists(IMG_PATH):
        os.mkdir(IMG_PATH)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    plot_model(model, to_file=f"{new_path}/model_arch_v1.png", show_shapes=True)


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
