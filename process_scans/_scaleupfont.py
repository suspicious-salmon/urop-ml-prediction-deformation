import cv2
import math

import _cvutil

# scale factor to scale up 80x80px nominal images to the scale of Sara's CT scans.
# SF = 17.103

# scale factor to scale down cad images to the scale of my CT scans.
# my scans = 9.8px/mm
# cad images = 85.515px/mm (== px/mm of Sara's CT scans)
SF = 0.1146

def pad_image(image, target_width, target_height, pad_value=0):
    """Pads or crops grayscale image with equal amount of pixels on top & bottom and on left & right to match targets.

    Args:
        image (numpy array): grayscale opencv image to be padded/cropped
        target_width (int): target width
        target_height (int): target height
        pad_value (int, optional): intensity value (0-255) to fill padded area with. Defaults to 0.

    Returns:
        numpy array: padded/cropped opencv image
    """
    pad_vertical = target_height-image.shape[0]
    if pad_vertical > 0:
        image = cv2.copyMakeBorder(image,
            math.floor(pad_vertical/2),
            math.ceil(pad_vertical/2),
            0, 0, cv2.BORDER_CONSTANT, value=pad_value)
    elif pad_vertical < 0:
        image = image[math.floor(-pad_vertical/2):-math.ceil(-pad_vertical/2),:]

    pad_horizontal = target_width-image.shape[1]
    if pad_horizontal > 0:
        image = cv2.copyMakeBorder(image,
            0, 0,
            math.floor(pad_horizontal/2),
            math.ceil(pad_horizontal/2),
            cv2.BORDER_CONSTANT, value=pad_value)
    elif pad_horizontal < 0:
        image = image[:,math.floor(-pad_horizontal/2):-math.ceil(-pad_horizontal/2)]
        
    return image

def scaleup(in_img_dir: str, out_img_dir: str, target_width: int, target_height: int, scale_factor=SF, invert=True):
    """Performs these operations:
    - inverts grayscale image from `in_img_dir`
    - thresholds to keep highest half of intensities
    - resizes by `scale factor`
    - pads to match `target_width` and `target_height`
    - saves result to `out_img_dir`
    """

    img = _cvutil.readim(in_img_dir, cv2.IMREAD_GRAYSCALE)[:,80:]
    if invert: img = 255-img
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
    img = cv2.resize(img, (int(round(SF*img.shape[1])), int(round(SF*img.shape[0]))), interpolation=cv2.INTER_NEAREST)
    img = pad_image(img, target_width, target_height, 0)

    _cvutil.writeim(out_img_dir, img)