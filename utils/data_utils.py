import os

import cv2


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
