import cv2
import numpy as np

import _cvutil

def heatmap(img_cad_dir, img_scan_dir, out_dir):
    """Creates a "heatmap" image comparing the differences between black-and-white images from img_cad_dir and img_scan_dir.
    Pixels which are white in both are set to gray.
    Pixels which are black in both are set to white. (confusing right?)
    Pixels which are white in img_cad and black in img_scan are set to red (i.e. "lost" material in the scan)
    Pixels which are black in img_cad and white in img_scan are set to blue (i.e. "gained" material in the scan)

    Args:
        img_cad_dir (directory string): directory of img_cad
        img_scan_dir (directory string): directory of img_scan
        out_dir (directory string): directory to save heatmap to
    """

    img_cad = _cvutil.readim(img_cad_dir, cv2.IMREAD_GRAYSCALE)
    img_scan = _cvutil.readim(img_scan_dir, cv2.IMREAD_GRAYSCALE)

    # make images black-and-white
    img_cad = cv2.threshold(img_cad, 20, 255, cv2.THRESH_BINARY)[1]
    img_scan = cv2.threshold(img_scan, 20, 255, cv2.THRESH_BINARY)[1]

    lost_material = np.logical_and(img_cad == 255, img_scan == 0).astype(int)
    gained_material = np.logical_and(img_cad == 0, img_scan == 255).astype(int)

    heatmap = cv2.cvtColor(255-img_cad, cv2.COLOR_GRAY2RGB)
    heatmap[img_cad == 255] = (200,200,200) # set pixels in cad to gray (lost pixels will be overwritten next)
    heatmap[lost_material == 1] = (0,0,255) # set lost pixels to red
    heatmap[gained_material == 1] = (255,0,0) # set gained pixels to blue

    _cvutil.writeim(out_dir, heatmap)