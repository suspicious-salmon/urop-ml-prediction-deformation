import cv2
import math

import _cvutil

# scale factor to scale up 80x80px nominal images to the scale of the CT scans.
SF = 17.103

def pad_image(image, target_width, target_height, pad_value=0):
    """Pads grayscale image with equal amount of pixels on top & bottom and on left & right.

    Args:
        image (numpy array): grayscale opencv image to be padded
        target_width (int): target width
        target_height (int): target height
        pad_value (int, optional): intensity value (0-255) to fill padded area with. Defaults to 0.

    Returns:
        numpy array: padded opencv image
    """
    pad_vertical = target_height-image.shape[0]
    pad_horizontal = target_width-image.shape[1]
    image = cv2.copyMakeBorder(image,
                               math.floor(pad_vertical/2),
                               math.ceil(pad_vertical/2),
                               math.floor(pad_horizontal/2),
                               math.ceil(pad_horizontal/2),
                               cv2.BORDER_CONSTANT, value=pad_value)
    return image

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
    img = pad_image(img, target_width, target_height, 0)

    _cvutil.writeim(out_img_dir, img)