import cv2
import numpy as np
import math
from scipy.optimize import minimize_scalar

import _cvutil

def align_ccorr(cad_dir, scan_dir, output_dirs, angle_bounds=(-180,180)):
    """Rotates and translates (black-and-white, i.e. 0 or 255) image at directory scan_dir to match image at directory cad_dir by minimising cross-correlation. Also saves:
    - missing pixels image: after alignment, map of pixels which are white in image at cad_dir but black in image at scan_dir
    - extra pixels image: after alignment, map of pixels which are white in image at scan_dir but black in image at cad_dir
    Images at scan_dir and cad_dir must be the same dimensions.

    Args:
        cad_dir (directory string): directory of cad image
        scan_dir (directory string): directory of scan image
        output_dirs (dict): output directories dictionary {"aligned" : directory string to write aligned image from scan_dir,
            "extra_pixels" : directory string to write missing pixels image,
            "missing_pixels" : directory string to write extra pixels image}
        angle_bounds (tuple): (min angle, max angle) range of angles in degrees within which to search for minium cross-correlation. Defaults to plusminus 180 degrees.
    """

    cad_img = _cvutil.readim(cad_dir, cv2.IMREAD_GRAYSCALE)
    scan_img = _cvutil.readim(scan_dir, cv2.IMREAD_GRAYSCALE)

    assert cad_img.shape == scan_img.shape, "Cad image and scan image must have the same width and height"

    # pad cad image and scan image to allow rotation at any angle without clipping the content. there is enough padding to fully contain a circle whose radius is the distance between the centre and a corner of the unpadded image.

    diameter = math.ceil(math.sqrt(cad_img.shape[0]**2 + cad_img.shape[1]**2))

    cad_img_padded = np.zeros((diameter, diameter), dtype = cad_img.dtype)
    cad_shift_rows = int((cad_img_padded.shape[0] - cad_img.shape[0])/2)
    cad_shift_cols = int((cad_img_padded.shape[1] - cad_img.shape[1])/2)
    cad_img_padded[cad_shift_rows:cad_shift_rows+cad_img.shape[0], cad_shift_cols:cad_shift_cols+cad_img.shape[1]] = cad_img

    scan_img_padded = np.zeros((diameter, diameter), dtype = scan_img.dtype)
    scan_shift_rows = int((scan_img_padded.shape[0] - scan_img.shape[0])/2)
    scan_shift_cols = int((scan_img_padded.shape[1] - scan_img.shape[1])/2)
    scan_img_padded[scan_shift_rows:scan_shift_rows+scan_img.shape[0], scan_shift_cols:scan_shift_cols+scan_img.shape[1]] = scan_img

    # find angle within angle_bounds that gives lowest minimum cross-correlation between scan_img_padded cad_img_padded

    def ccorr(rot: float):
        # rotate cad_img_padded by rot degrees
        rows, cols = scan_img_padded.shape
        M = cv2.getRotationMatrix2D(center=((cols-1)/2.0,(rows-1)/2.0), angle=rot, scale=1)
        im_rot = cv2.warpAffine(scan_img_padded, M, dsize=(cols,rows))
        # crop the template
        im_rot = im_rot[scan_shift_rows:scan_shift_rows+scan_img.shape[0], scan_shift_cols:scan_shift_cols+scan_img.shape[1]]

        result = cv2.matchTemplate(cad_img_padded, im_rot, cv2.TM_CCOEFF)
        return result

    res = minimize_scalar(lambda x: -ccorr(x).max(), bounds=angle_bounds, method='bounded')
    result = ccorr(res.x)

    # rotate and translate scan_img to minimum cross-correlation

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    rows, cols = cad_img_padded.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), res.x, 1)
    scan_img_aligned = cv2.warpAffine(scan_img_padded, M, (scan_img_padded.shape[1], scan_img_padded.shape[0]))
    scan_img_aligned = np.roll(scan_img_aligned, maxLoc[1] - scan_shift_rows, axis=0)
    scan_img_aligned = np.roll(scan_img_aligned, maxLoc[0] - scan_shift_cols, axis=1)

    scan_out = scan_img_aligned[cad_shift_rows:cad_shift_rows+cad_img.shape[0], cad_shift_cols:cad_shift_cols+cad_img.shape[1]]

    # crop out unimportant bits
    kernel_size = 300
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cad_img_inflated = cv2.dilate(cad_img, kernel)
    scan_out[cad_img_inflated == 0] = 0

    # write images

    _cvutil.writeim(output_dirs["aligned"], scan_out)

    extra_pixels = scan_out.copy()
    extra_pixels[cad_img != 0] = 0
    _cvutil.writeim(output_dirs["extra_pixels"], extra_pixels)

    missing_pixels = cad_img.copy()
    missing_pixels[scan_out > 20] = 0
    _cvutil.writeim(output_dirs["missing_pixels"], missing_pixels)

def crop_to_inflated_cad(cad_dir, img_dir, out_dir, kernel_size=300, overwrite=False):
    """Inflates (i.e. dilates) image from cad_dir, then sets all pixels in image from img_dir that are zero in inflated cad to zero and saves the result in out_dir.

    Args:
        cad_dir (directory string): directory of cad image to be inflated
        img_dir (directory string): directory of image to be cropped
        out_dir (directory string): directory to save cropped image to
        kernel_size (int, optional): kernel size used to dilate image from cad_dir. Defaults to 300.
        overwrite (bool, optional): if False, will raise OSError if a file at out_dir already exists. Defaults to False.
    """

    cad_img = _cvutil.readim(cad_dir, cv2.IMREAD_GRAYSCALE)
    img = _cvutil.readim(img_dir, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cad_img_inflated = cv2.dilate(cad_img, kernel)
    img[cad_img_inflated == 0] = 0

    _cvutil.writeim(out_dir, img, overwrite)