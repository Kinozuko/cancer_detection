import argparse
import sys
import time
import warnings

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils.constants import METRICS, MODEL_VERSIONS
from utils.data_utils import train_test_as_tensor
from utils.model_utils import (
    evaluate_model,
    generate_predictions,
    plot_confusion_matrix,
    plot_log_loss,
    plot_metrics,
    plot_roc_curves,
    save_model_info,
    save_precision_results,
)
from utils.models import get_model
from utils.process_image import process_images, read_images_dataset


def run_process_images(n_pools: int = 2):
    start_time = time.time()
    process_images(n_pools=2)
    end_time = time.time()

    print(f"Process Images runs in {end_time-start_time} seconds")


def run_train_model(
    version: str = "v1", n_batch: int = 30, n_epoch: int = 1, patience: int = 5
):
    start_time = time.time()
    x, y = read_images_dataset()

    train_ds, test_ds, validation_ds, train_test_np = train_test_as_tensor(x, y)

    model = get_model(version)

    if model:

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=1e-3, beta_1=0.92, beta_2=0.999),
            metrics=METRICS,
        )

        early_callback = EarlyStopping(
            monitor="val_auc",
            verbose=1,
            patience=patience,
            mode="max",
            restore_best_weights=True,
        )

        history = model.fit(
            train_ds.batch(batch_size=n_batch),
            epochs=n_epoch,
            validation_data=validation_ds.batch(batch_size=n_batch),
            callbacks=[early_callback],
        )

        model = save_model_info(model, early_callback, version=version)

        plot_log_loss(history, f"Model {version}", version)
        plot_metrics(history, version)
        evaluate_model(model, test_ds, version, batch_size=n_batch)

        y_train_pred, y_test_pred = generate_predictions(
            model, train_test_np["train"][0], train_test_np["test"][0]
        )

        plot_confusion_matrix(
            train_test_np["train"][1], y_train_pred, version=version, y_type="train"
        )
        plot_confusion_matrix(
            train_test_np["test"][1], y_test_pred, version=version, y_type="test"
        )

        save_precision_results(
            train_test_np["train"][1], y_train_pred, version=version, y_type="train"
        )
        save_precision_results(
            train_test_np["test"][1], y_test_pred, version=version, y_type="test"
        )

        plot_roc_curves(
            [
                {
                    "name": "Train Base",
                    "labels": train_test_np["train"][1],
                    "predictions": y_train_pred,
                },
                {
                    "name": "Test Base",
                    "labels": train_test_np["test"][1],
                    "predictions": y_test_pred,
                },
            ],
            version,
        )

        tf.keras.backend.clear_session()

        end_time = time.time()

        print(f"Training model runs in {end_time-start_time} seconds")
    else:
        print(f"There's no model with version {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        default="process",
        choices=["process", "train"],
        help="Type of action to execute the script",
    )
    parser.add_argument(
        "--n-pools", type=int, default=2, help="Number of parallel processes to run"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help=f"Version of the model to train {MODEL_VERSIONS}",
    )
    parser.add_argument(
        "--n-batch", type=int, default=30, help=f"Number of batches to use"
    )
    parser.add_argument(
        "--n-epoch", type=int, default=1, help=f"Number of batches to use"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help=f"Patience to use in Early Stopping"
    )
    args = parser.parse_args()

    if args.method == "process":
        if int(args.n_pools) <= 0:
            print(f"n_pools argument need to be greater to 0")
            sys.exit()
        run_process_images(n_pools=args.n_pools)
    elif args.method == "train":
        if (
            args.model_version not in MODEL_VERSIONS
            and parser.n_batch <= 0
            and parser.n_epoch <= 0
            and parser.patience <= 0
        ):
            print(f"Some arguments are wrong")
            sys.exit()
        run_train_model(args.model_version, args.n_batch, args.n_epoch, args.patience)
