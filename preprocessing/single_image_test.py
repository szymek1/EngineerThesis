import numpy as np
import cv2 as cv
import pathlib
from sklearn.cluster import MeanShift, estimate_bandwidth
from typing import Generator

from utils.preprocess import Preprocessor
from utils.performance_metric import timeit


@timeit
def apply_ms(imgs: Generator[pathlib.PosixPath, None, None], destination: str) -> None:
    """
    Applies Mean Shift to every large resolution image
    :param imgs: generator to images
    :param destination: destination path where to save images
    :return: nothing
    """
    img_count = 0
    name = "sample"

    for img in imgs:
        img = cv.imread(str(img))
        img_copy = img.copy()
        img = cv.medianBlur(img, 3)
        flat_image = img.reshape((-1, 3))
        flat_image = np.float32(flat_image)
        bandwidth = estimate_bandwidth(flat_image, quantile=0.3, n_samples=2000)
        ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
        ms.fit(flat_image)
        labeled = ms.labels_
        segments = np.unique(labeled)

        total = np.zeros((segments.shape[0], 3), dtype=float)
        count = np.zeros(total.shape, dtype=float)
        for i, label in enumerate(labeled):
            total[label] = total[label] + flat_image[i]
            count[label] += 1
        avg = total / count
        avg = np.uint8(avg)

        res = avg[labeled]
        result = res.reshape(img_copy.shape)
        cv.imwrite(str(pathlib.PurePath(destination).joinpath(name + str(img_count) + '.png')), result)
        img_count += 1


if __name__ == "__main__":
    img_dir = ["/home/szymonbogus/Documents/myProjects/STUDIA/inzynierka/temp_imgs/ProcessedImages/org_imgs"]
    dest_dir = "/home/szymonbogus/Documents/myProjects/STUDIA/inzynierka/temp_imgs/ProcessedImages/Single_q03_n2000"
    processor = Preprocessor(img_dir, dest_dir)
    all_imgs = processor.get_images()
    apply_ms(imgs=all_imgs, destination=dest_dir)
