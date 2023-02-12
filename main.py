import argparse
import sys
import time

from utils.constants import MODEL_VERSIONS
from utils.data_utils import train_test_as_tensor
from utils.models import get_model
from utils.process_image import process_images, read_images_dataset


def run_process_images(n_pools: int = 2):
    start_time = time.time()
    process_images(n_pools=2)
    end_time = time.time()

    print(f"Process Images runs in {end_time-start_time} seconds")


def run_train_model(version: str = "v1"):
    start_time = time.time()
    x, y = read_images_dataset()
    train_ds, test_ds, validation_ds = train_test_as_tensor(x, y)
    model = get_model(version)
    end_time = time.time()

    print(f"Training model runs in {end_time-start_time} seconds")


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
        help=f"Version of the model to train {MODEL_VERSIONS}",
    )
    args = parser.parse_args()

    if args.method == "process":
        if int(args.n_pools) <= 0:
            print(f"n_pools argument need to be greater to 0")
            sys.exit()
        run_process_images(n_pools=args.n_pools)
    elif args.method == "train":
        if args.model_version not in MODEL_VERSIONS:
            print(f"model_version argument need to be one this: {MODEL_VERSIONS}")
            sys.exit()
        run_train_model(args.model_version)
