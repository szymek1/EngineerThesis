"""
Set of utilities necessary for preprocessing images for fitting into DL models
"""

import pathlib
from pathlib import Path
from typing import List, Tuple, Generator

import cv2

from .IProcessing import IProcess


class Preprocessor(IProcess):
    """
    Class containing utilities for cropping images and smoothening them
    """

    def __init__(self, img_dirs: List[str], destination_path: str) -> None:
        """
        :param img_dirs:
        - imgs_dirs: set of paths to images which are to be preprocessed
        - destination_path: path to a directory where images are saved to
        """

        self._imgPaths: List[str] = img_dirs
        self._destPath: str = destination_path
        self._imgExtensions: List[str] = ['.jpg', '.jpeg', '.png']
        # self._imgs_lists: List[List[str]] = list()

    def change_resolution(self, new_resolution: Tuple[int, int]):
        """
        :param new_resolution:
        - new_resolution: format (int, int), changes resolution of found images
        :return:
        """

        for i in self.get_images():
            image_path = i
            image_name = str(image_path.name)
            roi = cv2.imread(str(image_path))
            roi = roi[0:new_resolution[1], 0:new_resolution[0]]
            cv2.imwrite(str(pathlib.PurePath(self._destPath).joinpath(image_name)), roi)

    def get_images(self) -> Generator[pathlib.PosixPath, None, None]:
        """
        :return: list of all images found in given directories
        """

        # found_imgs: List[List[str]] = list()

        for directory in self._imgPaths:
            for path in Path(directory).rglob('*'):
                if path.suffix.lower() in self._imgExtensions:
                    yield path
        """
                for directory in self._imgPaths:
            sub_list = list()
            for path in Path(directory).rglob('*'):
                if path.suffix.lower() in self._imgExtensions:
                    sub_list.append(str(path))
            found_imgs.append(sub_list)

        return found_imgs
        """
