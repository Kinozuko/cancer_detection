from tensorflow.keras import Model
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import (
    BatchNormalization,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

from .constants import SHAPE
from .model_utils import save_img_model


def model_version_v1(save_fig: bool = True):
    input_layer = Input(shape=SHAPE)
    conv1_layer = Convolution2D(
        16, (5, 5), padding="same", kernel_regularizer=l2(0.001), activation=relu
    )(input_layer)
    conv2_layer = Convolution2D(
        8, (3, 3), padding="same", kernel_regularizer=l2(0.001), activation=relu
    )(conv1_layer)
    maxpool1_layer = MaxPool2D(pool_size=(2, 2))(conv2_layer)
    norm1_layer = BatchNormalization()(maxpool1_layer)

    flat1_layer = Flatten()(norm1_layer)
    drop1_layer = Dropout(0.5)(flat1_layer)
    pred_layer = Dense(1, kernel_initializer=GlorotNormal(), activation=sigmoid)(
        drop1_layer
    )

    model = Model(inputs=input_layer, outputs=pred_layer)

    if save_fig:
        save_img_model(model)

    return model


def get_model(version: str = "v1", save_fig: bool = True):
    try:
        version_to_function = {"v1": model_version_v1}
        return version_to_function[version]()
    except KeyError as e:
        print(e)
        return None
