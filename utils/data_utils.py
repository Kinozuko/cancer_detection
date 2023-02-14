import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset


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

    return (
        train_ds,
        test_ds,
        validation_ds,
        {
            "train": (x_train, y_train),
            "test": (x_test, y_test),
        },
    )
