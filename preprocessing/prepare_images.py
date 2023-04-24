"""
Script activating processing pipeline
"""

import argparse
from typing import Tuple

from utils.preprocess import Preprocessor, MSProcessor


def tuple_of_ints(s) -> Tuple[int]:
    try:
        return tuple(int(x) for x in s.split(','))
    except argparse.ArgumentTypeError:
        print('Invalid tuple of integers')


def main(my_args: argparse.Namespace) -> None:
    imgs_directories = my_args.dirs
    save_destination = my_args.dest
    new_resolution = my_args.res
    use_ms = my_args.ms_use

    processor = Preprocessor(imgs_directories, save_destination)

    # TODO: make more clear !!!
    if new_resolution is not None:
        if use_ms is True:
            print("Using mean shift processor")
            processor = MSProcessor(imgs_directories, save_destination, 0.06, 100)
            processor.ms_cluster(False, new_resolution)
        else:
            processor.change_resolution(new_resolution)
    else:
        print("No resolution specified, default resolution will be used\nUsing mean shift processor")
        if use_ms is True:
            processor = MSProcessor(imgs_directories, save_destination, 0.01, 100)
            processor.ms_cluster(True)
        else:
            print("No action to take")
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch images processing pipeline",
                                     argument_default=argparse.SUPPRESS, allow_abbrev=False, add_help=False)
    parser.add_argument("--dirs", metavar='dir', help="paths to images directories", nargs='+')
    parser.add_argument("--dest", type=str, help="destination of processed images to be saved to")
    parser.add_argument("--res", metavar='tuple', type=tuple_of_ints, help="set new resolution", nargs='?')
    parser.add_argument("--ms_use", type=bool, help="cluster images using Mean Shift", nargs='?')
    parser.add_argument("-h", "--help", action="help", help="Display this message")

    args = parser.parse_args()
    main(args)
