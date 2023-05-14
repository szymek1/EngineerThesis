import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth


img = cv.imread('/home/szymonbogus/Documents/myProjects/STUDIA/inzynierka/temp_imgs/4/zdjecie_po_edycjia.png')

# filter to reduce noise
img = cv.medianBlur(img, 3)

# flatten the image
flat_image = img.reshape((-1,3))
flat_image = np.float32(flat_image)

# meanshift
bandwidth = estimate_bandwidth(flat_image, quantile=0.06, n_samples=1000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled=ms.labels_


# get number of segments
segments = np.unique(labeled)


# get the average color of each segment
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# cast the labeled image into the corresponding average color
res = avg[labeled]
result = res.reshape((img.shape))
cv.imwrite('/home/szymonbogus/Documents/myProjects/STUDIA/inzynierka/temp_imgs/test/sample4test_quantile006_nsamples1000.png', result)
# show the result
"""
cv.imshow('result',result)
cv.waitKey(0)
cv.destroyAllWindows()
"""
