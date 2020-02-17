import argparse

from reuters.main import main


def parse_args():
    """
    Parses input arguments to the program, can improve the encompass all important hyper parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to the dataset folder, containing raw (sgml) or processed files",
    )

    return parser.parse_known_args()[0]


if __name__ == "__main__":
    main(parse_args())
