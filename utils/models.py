from tensorflow.keras import Model, Sequential
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.applications import ResNet152
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
        save_img_model(model, version="v1")

    return model


def model_version_v2(save_fig: bool = True):
    base_model = ResNet152(
        include_top=False,
        weights="imagenet",
        input_shape=SHAPE,
        classes=2,
        classifier_activation="sigmoid",
    )

    base_model.trainable = False  # We keep the original weights

    model = Sequential(
        [
            base_model,
            Convolution2D(
                8, 8, padding="same", kernel_regularizer=l2(0.001), activation=relu
            ),
            MaxPool2D(pool_size=(2, 2)),
            Convolution2D(
                4, 4, padding="same", kernel_regularizer=l2(0.001), activation=relu
            ),
            Dropout(0.45),
            Convolution2D(
                2, 2, padding="same", kernel_regularizer=l2(0.001), activation=relu
            ),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),
            Flatten(),
            Dropout(0.45),
            Dense(1, kernel_initializer=GlorotNormal(), activation=sigmoid),
        ]
    )

    if save_fig:
        save_img_model(model, version="v2")

    return model


def get_model(version: str = "v1", save_fig: bool = True):
    try:
        version_to_function = {"v1": model_version_v1, "v2": model_version_v2}
        return version_to_function[version]()
    except KeyError as e:
        print(e)
        return None
