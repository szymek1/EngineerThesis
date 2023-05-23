"""
Script activating processing pipeline
"""

import argparse
from typing import Tuple, Union

from utils.preprocess import Preprocessor, MSProcessor
from utils.performance_metric import timeit


def tuple_of_ints(s) -> Union[Tuple[int], None]:
    try:
        return tuple(int(x) for x in s.split(','))
    except argparse.ArgumentTypeError:
        print("Argument type must be tuple of integers or do not pass anything")


@timeit
def main(my_args: argparse.Namespace) -> None:
    imgs_directories = my_args.dirs
    save_destination = my_args.dest
    try:
        new_resolution = my_args.res
    except AttributeError:
        new_resolution = None
    use_ms = my_args.ms_use

    if use_ms is True:
        processor = MSProcessor(imgs_directories, save_destination, 0.08, 50)
        if new_resolution is not None:
            print("Using mean shift processor...\nResolution will be changed...")
            processor.ms_cluster(True, new_resolution)
        else:
            print("Using mean shift processor...\nResolution will remain...")
            processor.ms_cluster(False)
    else:
        processor = Preprocessor(imgs_directories, save_destination)
        if new_resolution is not None:
            print("Resolution will be changed...\nNo Mean Shift processing applied...")
            processor.change_resolution(new_resolution)
        else:
            print("No action to take\nApp closes...")
            exit()


# TODO: Add tests into another directory
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch images processing pipeline",
                                     argument_default=argparse.SUPPRESS, allow_abbrev=False, add_help=False)
    parser.add_argument("--dirs", metavar='dir', help="paths to images directories", nargs='+')
    parser.add_argument("--dest", type=str, help="destination of processed images to be saved to")
    parser.add_argument("--res", metavar='tuple', type=tuple_of_ints, help="set new resolution", required=False)
    parser.add_argument("--ms_use", type=bool, help="cluster images using Mean Shift", required=False)
    parser.add_argument("-h", "--help", action="help", help="Display this message")

    args = parser.parse_args()
    main(args)
