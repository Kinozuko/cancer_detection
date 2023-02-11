import multiprocessing as mp
import os

import cv2
import dicomsdl
import numpy as np

from .data_utils import count_images_and_keep

RESIZE = (512, 512)  # Image resize
BASE_PATH = os.getcwd()
DATA_PATH = BASE_PATH + "/data"
TRAIN_PATH = DATA_PATH + "/train_images"
TEST_PATH = DATA_PATH + "/test_images"
FILEPATH_TO_WORK = [TRAIN_PATH, TEST_PATH]


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
