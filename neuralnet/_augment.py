"""This module is designed to apply random data augmentations. I had to write my own instead of using the built-in pytorch functions,
because I wanted to be able to apply the same augmentation to both the feature and label of my generative model
(i.e. both feature and label need to be transformed in the same way)."""

import cv2
import random
import math

def _pad_around_centre(image, target_width, target_heigh, pad_value=0):
    """Pads image with equal amount of pixels on top & bottom and on left & right."""
    pad_vertical = target_height-image.shape[0]
    pad_horizontal = target_width-image.shape[1]
    image = cv2.copyMakeBorder(image,
                               math.floor(pad_vertical/2),
                               math.ceil(pad_vertical/2),
                               math.floor(pad_horizontal/2),
                               math.ceil(pad_horizontal/2),
                               cv2.BORDER_CONSTANT, value=pad_value)
    return image

def _crop_to_centre(image, target_width, target_height):
    """Crops image to target dimensions around its centre."""
    h,w = image.shape
    crop_x = w - target_width
    crop_y = h - target_height
    image = image[crop_y//2:crop_y//2+target_height, crop_x//2:crop_x//2+target_width]
    return image
    
def _random_rotate(images, angle_bounds, flags, border_mode=cv2.BORDER_CONSTANT, fill=0):
    """Selects random rotation angle (in degrees) within angle_bounds. Applies this rotation to *every* image in images. Empty space left in corners after rotation is filled with fill."""
    angle = random.uniform(*angle_bounds)
    
    ret_images = [None for i in range(len(images))]
    for i in range(len(images)):
        h,w = images[i].shape[-2:]
        M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, 1)
        ret_images[i] = cv2.warpAffine(images[i], M, (h,w), flags=flags, borderMode=border_mode, borderValue=fill)

    return ret_images

def _random_scale(images, scale_bounds, fill=0):
    """Selects random scaling factor within scale_bounds. Applies this scaling factor to *every* image in images. Image is padded/cropped back to original dimensions."""
    scale = random.uniform(*scale_bounds)

    ret_images = [None for i in range(len(images))]
    for i in range(len(images)):
        h,w = images[i].shape[-2:]
        ret_images[i] = cv2.resize(images[i], (int(round(w*scale)), int(round(h*scale))))
        
        if scale < 1:
            ret_images[i] = _pad_around_centre(ret_images[i], w, h, fill)
        elif scale > 1:
            ret_images[i] = _crop_to_centre(ret_images[i], w, h)

    return ret_images

def _random_flip(images):
    """Selects 0 or 1 at random. If 1, *every* image in images is flipped vertically. If not, none are."""
    if random.randint(0,1) == 1:
        return [cv2.flip(image, 0) for image in images]
    else:
        return images

def deform1(in_images, background_value=0):
    """Applies a random rotation and bernoulli-random-variable-determined flip identically to all images in in_images.

    Args:
        images (iterable): list of *grayscale* opencv images (list of numpy arrays)
        background_value (int, optional): intensity value to fill space created when shrinking or rotating image. Defaults to 0.

    Returns:
        list of numpy arrays: Transformed images. Will be the same dimensions as they were input.
    """

    images = _random_flip(in_images)
    images = _random_rotate(in_images, (-180,180), flags=cv2.INTER_NEAREST, fill=background_value)

    return images