"""
Set of utilities necessary for preprocessing images for fitting into DL models
"""

import pathlib
from pathlib import Path
from typing import List, Tuple, Generator, Optional

import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler

from .IProcessing import IProcess


class Preprocessor(IProcess):
    """
    Class containing utilities for cropping images and smoothening them
    """

    def __init__(self, img_dirs: List[str], destination_path: str) -> None:
        """
        :param img_dirs: set of paths to images which are to be preprocessed
        :param destination_path: path to a directory where images are saved to
        """

        self._imgPaths: List[str] = img_dirs
        self._destPath: str = destination_path
        self._imgExtensions: List[str] = ['.jpg', '.jpeg', '.png']
        # self._imgs_lists: List[List[str]] = list()

    def change_resolution(self, new_resolution: Tuple[int, int]) -> None:
        """
        :param new_resolution: format (int, int), changes resolution of found images
        :return: None, only saves to given directory
        """

        for i in self.get_images():
            image_path = i
            image_name = str(image_path.name)
            roi = cv2.imread(str(image_path))
            roi = roi[0:new_resolution[1], 0:new_resolution[0]]
            cv2.imwrite(str(pathlib.PurePath(self._destPath).joinpath(image_name)), roi)

    def get_images(self) -> Generator[pathlib.PosixPath, None, None]:
        """
        :return: generator to found images
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


class MSProcessor(Preprocessor):
    """
    Class extensioning Preprocessor capabilities with Mean Shift Clustering
    """

    def __init__(self, img_dirs: List[str], destination_path: str, qunatile: float, samples_numb: int) -> None:
        """
        :param img_dirs: set of paths to images which are to be preprocessed
        :param destination_path: destination_path: path to a directory where images are saved to
        :param qunatile: distance between points used for estimating bandwidth
        :param samples_numb: subset of samples from a given region used for estimating bandwidth
        """

        super().__init__(img_dirs, destination_path)
        self._quantile: float = qunatile
        self._samples_numb: int = samples_numb

    def change_resolution(self, new_resolution: Tuple[int, int]) -> Generator[np.ndarray, None, None]:
        """
        :param new_resolution: format (int, int), changes resolution of found images
        :return: generator to cropped images, as numpy arrays
        """

        for i in self.get_images():
            image_path = i
            roi = cv2.imread(str(image_path))
            roi = roi[0:new_resolution[1], 0:new_resolution[0]]
            yield roi

    def ms_cluster(self, change_res: bool, new_resolution: Optional[Tuple[int, int]] = None) -> None:
        """
        :param change_res: checks if there is a need to call change_resolution
        :param new_resolution: format (int, int), changes resolution of found images
        :return: None, saves images to given destination
        """

        # TODO: find a way to not initialize mean shift instance every image !!!
        # TODO: fix bug- could not find a writer for the specified extension in function 'imwrite_' in cv2.imwrite()
        if change_res is True and new_resolution is None:
            raise ValueError("ms_cluster must have new_resolution set to not None if change_res == True!")
        if change_res is True:
            scaler = StandardScaler()

            for i in self.get_images():
                image_name = str(i.name)
                i = cv2.imread(str(i))
                pixels = np.reshape(i, (-1, 3))
                pixels = scaler.fit_transform(pixels)
                bandwidth = estimate_bandwidth(pixels, quantile=self._quantile, n_samples=self._samples_numb)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(pixels)
                labels = ms.labels_
                labels = np.reshape(labels, i.shape[:2])
                cv2.imwrite(str(pathlib.PurePath(self._destPath).joinpath(image_name)), labels)

        else:
            scaler = StandardScaler()

            iteration = 0
            name = "sample"

            for i in self.change_resolution(new_resolution):
                pixels = np.reshape(i, (-1, 3))
                pixels = scaler.fit_transform(pixels)
                bandwidth = estimate_bandwidth(pixels, quantile=self._quantile, n_samples=self._samples_numb)
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(pixels)
                labels = ms.labels_
                labels = np.reshape(labels, i.shape[:2])
                cv2.imwrite(str(pathlib.PurePath(self._destPath).joinpath(name + str(iteration))), labels)
                iteration += 1

