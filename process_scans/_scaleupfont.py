import cv2

import _cad
import _cvutil

SF = 17.103

def scaleup(in_img_dir: str, out_img_dir: str, target_width: int, target_height: int, scale_factor=SF):
    """Performs these operations:
    - inverts grayscale image from `in_img_dir`
    - thresholds to keep highest half of intensities
    - resizes by `scale factor`
    - pads to match `target_width` and `target_height`
    - saves result to `out_img_dir`
    """

    img = _cvutil.readim(in_img_dir, cv2.IMREAD_GRAYSCALE)[:,80:]
    img = 255-img
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
    img = cv2.resize(img, (int(round(SF*img.shape[1])), int(round(SF*img.shape[0]))), interpolation=cv2.INTER_NEAREST)
    img = _cad.pad_image(img, target_width, target_height, 0)

    _cvutil.writeim(out_img_dir, img)