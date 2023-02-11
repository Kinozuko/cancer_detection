import argparse
import sys
import time

from utils.process_image import process_images, read_images_dataset


def run_process_images(n_pools: int = 2):
    start_time = time.time()
    process_images(n_pools=2)
    end_time = time.time()

    print(f"Process Images runs in {end_time-start_time} seconds")


def run_train_model():
    start_time = time.time()
    x, y = read_images_dataset()

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
        "--n_pools", type=int, default=2, help="Number of parallel processes to run"
    )
    args = parser.parse_args()

    if args.method == "process":
        if int(args.n_pools) <= 0:
            print(f"n_pools argument need to be greater to 0")
            sys.exit()
        run_process_images(n_pools=args.n_pools)
    elif args.method == "train":
        run_train_model()
