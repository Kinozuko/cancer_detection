import multiprocessing as mp
import os

import cv2
import dicomsdl
import numpy as np
import pandas as pd

from .constants import (
    BASE_PATH,
    DATA_PATH,
    FILEPATH_TO_WORK,
    RESIZE,
    TEST_PATH,
    TRAIN_PATH,
)
from .data_utils import count_images_and_keep


def dicom_to_array(path: str):
    dcm_file = dicomsdl.open(path)  # Read the dcm file
    data = dcm_file.pixelData()  # Extract the pixel data

    data = (data - data.min()) / (data.max() - data.min())  # Normalize

    if dcm_file.getPixelDataInfo()["PhotometricInterpretation"] == "MONOCHROME1":
        # Rever grayscale if needed
        data = 1 - data

    data = cv2.resize(data, RESIZE)  # Resize the original dcm image
    data = (data * 255).astype(np.uint8)  # We multiply by the number of pixel to use
    return data


def reshape_to_png(path: str, files: list):
    success = 0
    errors = 0
    errors_string = []

    new_images_filepath = []

    for image in files:
        try:
            patient = image.split("/")[-2]  # Extract id patient
            image_id = image.split("/")[-1].split(".")[0]  # Extract image id

            patient_folder = path + "/" + patient  # New patient folder path
            if not os.path.exists(patient_folder):
                # Create folder if this not exists
                os.mkdir(patient_folder)
                pass

            image_filepath = patient_folder + "/" + image_id + ".png"  # New image name
            if not os.path.exists(image_filepath):
                # Create the image if this not exists
                cv2.imwrite(image_filepath, dicom_to_array(image))
                new_images_filepath.append(image_filepath)
                success += 1
                pass
        except Exception as e:
            errors_string.append(e)
            errors += 1

    print(f"Successful converted dcm to png with a size of {RESIZE}: {success}")
    print(f"Errors when converting dcm to png with a size of {RESIZE}: {errors}")
    return new_images_filepath, success, errors, errors_string


def process_images(n_pools: int = 2):
    if os.path.exists(DATA_PATH):
        path_dict = count_images_and_keep(FILEPATH_TO_WORK)

        train_images = path_dict[FILEPATH_TO_WORK[0]]
        test_images = path_dict[FILEPATH_TO_WORK[1]]

        new_train_folder = DATA_PATH + f"/train_{RESIZE[0]}_{RESIZE[1]}"
        new_test_folder = DATA_PATH + f"/test_{RESIZE[0]}_{RESIZE[1]}"

        print(f"There are {len(train_images)} train images")
        print(f"There are {len(test_images)} test images")

        if not os.path.exists(new_train_folder):
            print(f"New folder created: /{new_train_folder.split('/')[-1]}")
            os.mkdir(new_train_folder)

        if not os.path.exists(new_test_folder):
            print(f"New folder created: /{new_test_folder.split('/')[-1]}")
            os.mkdir(new_test_folder)

        with mp.Pool(n_pools) as p:
            p.starmap(
                reshape_to_png,
                [(new_train_folder, train_images), (new_test_folder, test_images)],
            )
    else:
        print(
            "To work with this script you need a data folder like we describe in the README.md"
        )


def read_images_dataset(type="parallel"):
    if type == "parallel":
        return read_images_dataset_parallel()
    else:
        return read_images_dataset_normal


def read_images_dataset_normal():
    data = []

    train_folder = (
        DATA_PATH + f"/train_{RESIZE[0]}_{RESIZE[1]}"
    )  # The folder is the same created int process_images function
    train_csv = pd.read_csv(DATA_PATH + "/train.csv")

    for patient_id in os.listdir(train_folder):

        patient_folder = train_folder + "/" + patient_id

        images_ids = [
            int(image.split(".png")[0]) for image in os.listdir(patient_folder)
        ]
        is_cancer = (
            train_csv[train_csv["image_id"].isin(images_ids)][["cancer", "image_id"]]
            .set_index("image_id")
            .to_dict(orient="index")
        )

        data.extend(
            list(
                map(
                    lambda s: (
                        cv2.imread(patient_folder + "/" + s),
                        is_cancer[int(s.split(".png")[0])]["cancer"],
                    ),
                    os.listdir(patient_folder),
                )
            )
        )

    # Now let's format the data to a numpy array

    x, y = [], []

    for raw_data in data:
        x.append(raw_data[0])
        y.append(raw_data[1])

    x_data = np.array(x)
    y_data = np.array(y)

    return x_data, y_data


def load_image(image_path, is_cancer):
    img = cv2.imread(image_path)
    return img, is_cancer


def read_images_dataset_parallel():
    data = []
    pool = mp.Pool()

    train_folder = os.path.join(DATA_PATH, f"train_{RESIZE[0]}_{RESIZE[1]}")
    train_csv = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

    for patient_id in os.listdir(train_folder):
        patient_folder = os.path.join(train_folder, patient_id)
        images_ids = [
            int(image.split(".png")[0]) for image in os.listdir(patient_folder)
        ]
        is_cancer = (
            train_csv[train_csv["image_id"].isin(images_ids)][["cancer", "image_id"]]
            .set_index("image_id")
            .to_dict(orient="index")
        )

        image_paths = [
            os.path.join(patient_folder, image) for image in os.listdir(patient_folder)
        ]
        results = pool.starmap(
            load_image,
            [
                (
                    image_path,
                    is_cancer[int(image_path.split("/")[-1].split(".png")[0])][
                        "cancer"
                    ],
                )
                for image_path in image_paths
            ],
        )
        data.extend(results)

    pool.close()
    pool.join()

    x_data = np.array([result[0] for result in data])
    y_data = np.array([result[1] for result in data])

    return x_data, y_data
