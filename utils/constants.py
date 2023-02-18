import os

from tensorflow.keras.metrics import (
    AUC,
    BinaryAccuracy,
    FalseNegatives,
    FalsePositives,
    Precision,
    Recall,
    SpecificityAtSensitivity,
    TrueNegatives,
    TruePositives,
)

MODEL_VERSIONS = ["v1", "v2"]
SHAPE = (512, 512, 3)
RESIZE = (512, 512)
BASE_PATH = os.getcwd()
DATA_PATH = BASE_PATH + "/data"
IMG_PATH = BASE_PATH + "/img"
INFO_PATH = BASE_PATH + "/info"
TRAIN_PATH = DATA_PATH + "/train_images"
TEST_PATH = DATA_PATH + "/test_images"
FILEPATH_TO_WORK = [TRAIN_PATH, TEST_PATH]
METRICS = [
    TruePositives(name="tp"),
    FalsePositives(name="fp"),
    TrueNegatives(name="tn"),
    FalseNegatives(name="fn"),
    BinaryAccuracy(name="accuracy"),
    Precision(name="precision"),
    Recall(name="recall"),
    AUC(name="auc"),
    SpecificityAtSensitivity(sensitivity=0.8, name="sensitivity"),
]
