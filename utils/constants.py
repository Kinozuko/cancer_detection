import os

MODEL_VERSIONS = ["v1"]
SHAPE = (512, 512, 3)
RESIZE = (512, 512)
BASE_PATH = os.getcwd()
DATA_PATH = BASE_PATH + "/data"
TRAIN_PATH = DATA_PATH + "/train_images"
TEST_PATH = DATA_PATH + "/test_images"
FILEPATH_TO_WORK = [TRAIN_PATH, TEST_PATH]
