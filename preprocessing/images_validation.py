"""
Script for validating processed images with MSE and SSIM metrics
"""

import json
import argparse
import pathlib
from typing import Generator

import cv2
import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity

from utils.preprocess import Preprocessor

# TODO: implement Canny edge validation with OpenCV
def validate_processing(images: tuple, report_dest: str) -> None:
    """
    Uses MSE and SSIM metrics to validate images processing technique
    :param images: tuple with generators to original and processed images
    :param report_dest: path to the destination where report should be saved
    :return: none
    """

    mse_score = list()
    ssim_score = list()

    for org_img, proc_img in zip(images[0], images[1]):
        """
        images[0]- generator with original images
        images[1]- generator with processed images
        """

        org_img = cv2.imread(str(org_img))
        proc_img = cv2.imread(str(proc_img))

        mse = mean_squared_error(org_img, proc_img)
        ssim = structural_similarity(org_img, proc_img, multichannel=True)

        mse_score.append(mse)
        ssim_score.append(ssim)

    mse_avg_score = np.mean(mse_score)
    ssim_avg_score = np.mean(ssim_score)

    report_dict = {
        "Summary:": "Processed images with MSE and SSIM metrics\nLower MSE values indicate better similarity between "
                    "the original and processed images\nHigher SSIM values indicate better preservation of image "
                    "details",
        "MSE: ": mse_avg_score,
        "SSIM: ": ssim_avg_score
    }

    with open(report_dest + "/mse_ssim_report.json", "w") as report:
        json.dump(report_dict, report)


def main(my_args: argparse.Namespace) -> None:
    imgs_directories = my_args.dirs
    save_destination = my_args.dest

    original_img_dir = "org_imgs"
    proceeded_img_dir = "proc_img"

    original_img_dir_path = str(pathlib.Path(imgs_directories).joinpath(original_img_dir))
    proceeded_img_dir_path = str(pathlib.Path(imgs_directories).joinpath(proceeded_img_dir))

    processor_4_org_img = Preprocessor(original_img_dir_path)
    processor_4_proc_img = Preprocessor(proceeded_img_dir_path)

    all_org_imgs: Generator[pathlib.PosixPath, None, None] = processor_4_org_img.get_images()
    all_proc_imgs: Generator[pathlib.PosixPath, None, None] = processor_4_proc_img.get_images()
    images_generators = (all_org_imgs, all_proc_imgs)

    validate_processing(images_generators, save_destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch images validation pipeline",
                                     argument_default=argparse.SUPPRESS, allow_abbrev=False, add_help=False)
    parser.add_argument("--dirs", metavar='dir', help="paths to images directories", nargs='+')
    parser.add_argument("--dest", type=str, help="destination of results to be saved")
    parser.add_argument("-h", "--help", action="help", help="Display this message\nOriginal images have to be in the "
                                                            "directory called /org_imgs, while processed images in "
                                                            "/proc_imgs\nImages in both directories have to be in the "
                                                            "same order")

    args = parser.parse_args()
    main(args)
