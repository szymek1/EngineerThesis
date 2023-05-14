"""
Set of utilities necessary for preprocessing images for fitting into DL models
"""

import pathlib
import multiprocessing
from math import sqrt
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

    def __init__(self, img_dirs: List[str], destination_path: Optional[str]) -> None:
        """
        :param img_dirs: set of paths to images which are to be preprocessed
        :param destination_path: path to a directory where images are saved to
        """

        self._imgPaths: List[str] = img_dirs
        self._destPath: str = destination_path
        self._imgExtensions: List[str] = ['.jpg', '.jpeg', '.png']

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

        for directory in self._imgPaths:
            for path in Path(directory).rglob('*'):
                if path.suffix.lower() in self._imgExtensions:
                    yield path


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

        if change_res is True and new_resolution is None:
            raise ValueError("ms_cluster must have new_resolution set to not None if change_res == True!")
        if change_res is True:

            num_processes = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=num_processes)

            iteration = 0
            name = "sample"

            for i in self.change_resolution(new_resolution):
                roi_list: List[np.ndarray] = self.cut_into_smaller_imgs(i, new_width=80, new_height=76)
                ready_rois: List[np.ndarray] = pool.map(self.ms_2_roi, roi_list)
                ready_rois = self.merge_rois_into_image(ready_rois, new_resolution)
                cv2.imwrite(str(pathlib.PurePath(self._destPath).joinpath(name + str(iteration) + '.png')), ready_rois)
                iteration += 1

        # TODO: add here steps for processing with ms but without resizing
        else:
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

    @staticmethod
    def cut_into_smaller_imgs(img: np.ndarray, new_width: int, new_height: int, roi_number: int = 64) -> \
            List[np.ndarray]:
        """
        Divides a full scale image into set of smaller images. This method should be called only if it is intended to
        work with new resolution og 640x608- tested for this cause.
        :param img: full resolution image
        :param new_width: new width for a single roi
        :param new_height: new height for a single roi
        :param roi_number: number of rois
        :return: list of small rois, which overall create one image
        """

        roi_list: List[np.ndarray] = list()
        division_factor = int(sqrt(roi_number))

        for row in range(division_factor):
            for col in range(division_factor):
                y_start = row * new_height
                y_end = y_start + new_height
                x_start = col * new_width
                x_end = x_start + new_width
                roi_list.append(img[y_start:y_end, x_start:x_end])

        return roi_list

    @staticmethod
    def merge_rois_into_image(roi_list: List[np.ndarray], roi_number: int = 64) -> np.ndarray:
        """
        Merges rois into full scale image
        :param roi_list: list of all small rois that an image consists of
        :param roi_number: number of rois in an image
        :return: merged image
        """

        division_factor = 8  # int(sqrt(roi_number))- for some reason if this line which is basically 8 is active
        # Python throws: TypeError: must be real number, not tuple

        single_rows = [tuple(roi_list[i:i + division_factor]) for i in range(0, 64, 8)]  # the same thing happen here
        # for roi_number=64 idk why
        single_rows_merged = [np.hstack(row) for row in single_rows]

        return np.vstack(single_rows_merged)

    def ms_2_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Applies Mean Shift to a single roi out of a whole image
        :param roi: small piece of a full scale image
        :return: the same roi but processed by Mean Shift
        """
        
        img = roi.copy()
        roi = cv2.medianBlur(roi, 3)
        roi = roi.reshape((-1, 3))
        roi = np.float32(roi)

        bandwidth = estimate_bandwidth(roi, quantile=self._quantile, n_samples=self._samples_numb)
        ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
        ms.fit(roi)
        labeled = ms.labels_

        segments = np.unique(labeled)

        total = np.zeros((segments.shape[0], 3), dtype=float)
        count = np.zeros(total.shape, dtype=float)
        for i, label in enumerate(labeled):
            total[label] = total[label] + roi[i]
            count[label] += 1
        avg = total / count
        avg = np.uint8(avg)

        res = avg[labeled]
        result = res.reshape(img.shape)

        return result
