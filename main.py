import os
from utils.data_utils import count_images_and_keep
from utils.process_image import dicom_to_array

BASE_PATH = os.getcwd()
DATA_PATH = BASE_PATH + "/data"
TRAIN_PATH = DATA_PATH + "/train_images"
TEST_PATH = DATA_PATH + "/test_images"
FILEPATH_TO_WORK = [TRAIN_PATH, TEST_PATH]


def process_images():
  if os.path.exists(DATA_PATH):
    path_dict = count_images_and_keep(FILEPATH_TO_WORK)

    train_images = path_dict[FILEPATH_TO_WORK[0]]
    test_images = path_dict[FILEPATH_TO_WORK[1]]

    print(f"We have {len(train_images)} for train images")
    print(f"We have {len(test_images)} for train images")
  else:
      print("To work with this script you need a data folder like we describe in the README.ms")

if __name__=='__main__':
    process_images()
