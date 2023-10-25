"""This file, given a folder of black-and-white images (should be the thresholded outputs
of the trained NN), scales those down to 80x80px as in the original source fonts, at the correct scale factor.
These 80x80px outputs can then be converted into STL files to 3D print."""

import cv2
import os
import math
import matplotlib.pyplot as plt

SF = 17.103
original_dim = 80
scaled_dim = int(round(SF*original_dim))
input_dim = 2460
pad_each_side = (input_dim - scaled_dim)//2

in_dir = r"E:\greg\Chinese Characters\3D Printed Deformations\urop-structured-nn\other\ModelOutputThresholded"
files = os.scandir(in_dir)
for file in files:
    img = 255-cv2.imread(os.path.join(in_dir, file.name), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (input_dim, input_dim), interpolation=cv2.INTER_NEAREST)
    unpadded = img[pad_each_side:pad_each_side+scaled_dim,pad_each_side:pad_each_side+scaled_dim]
    print(f"Downscale SF will be {unpadded.shape[0]/original_dim}")
    downscaled = cv2.resize(unpadded, (original_dim, original_dim), interpolation=cv2.INTER_NEAREST)

    # plt.imshow(downscaled, cmap="gray")
    # plt.show()
    cv2.imwrite(f"Downscaled\\downscaled_{file.name}", downscaled)