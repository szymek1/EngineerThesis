"""
Script activating processing pipeline
"""

import argparse
from typing import Tuple

from utils.preprocess import Preprocessor


def tuple_of_ints(s) -> Tuple[int]:
    try:
        return tuple(int(x) for x in s.split(','))
    except argparse.ArgumentTypeError:
        print('Invalid tuple of integers')


def main(my_args: argparse.Namespace) -> None:
    imgs_directories = my_args.dirs
    save_destination = my_args.dest
    new_resolution = my_args.res

    processor = Preprocessor(imgs_directories, save_destination)

    if new_resolution is not None:
        processor.change_resolution(new_resolution)
    else:
        print("No resolution specified, default resolution will be used")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch images processing pipeline",
                                     argument_default=argparse.SUPPRESS, allow_abbrev=False, add_help=False)
    parser.add_argument("--dirs", metavar='dir', help="paths to images directories", nargs='+')
    parser.add_argument("--dest", type=str, help="destination of processed images to be saved to")
    parser.add_argument("--res", metavar='tuple', type=tuple_of_ints, help="set new resolution", nargs='?')
    parser.add_argument("-h", "--help", action="help", help="Display this message")

    args = parser.parse_args()
    main(args)
