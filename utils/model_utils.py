import os

from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from .constants import IMG_PATH, INFO_PATH


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
