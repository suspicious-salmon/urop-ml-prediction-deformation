import cv2
import random
import math

# Is there value in using translation as augmentation if we can already centre all the data to a centroid (i.e. real inputs will never have translation issues?)

def ensure_same_shape(images):
    shape = images[0].shape
    for image in images[1:]:
        assert image.shape == shape, "Images must be the same shape."

def pad_around_centre(image, target_width, target_height, pad_value=0):
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

def crop_to_centre(image, target_width, target_height):
    h,w = image.shape
    crop_x = w - target_width
    crop_y = h - target_height
    image = image[crop_y//2:crop_y//2+target_height, crop_x//2:crop_x//2+target_width]
    return image
    
def random_rotate(images, angle_bounds, flags, border_mode=cv2.BORDER_CONSTANT, fill=255):
    angle = random.uniform(*angle_bounds)
    
    ret_images = [None for i in range(len(images))]
    for i in range(len(images)):
        h,w = images[i].shape[-2:]
        M = cv2.getRotationMatrix2D(((w-1)/2.0, (h-1)/2.0), angle, 1)
        ret_images[i] = cv2.warpAffine(images[i], M, (h,w), flags=flags, borderMode=border_mode, borderValue=fill)

    return ret_images

def random_scale(images, scale_bounds, fill=255):
    scale = random.uniform(*scale_bounds)

    ret_images = [None for i in range(len(images))]
    for i in range(len(images)):
        h,w = images[i].shape[-2:]
        ret_images[i] = cv2.resize(images[i], (int(round(w*scale)), int(round(h*scale))))
        
        if scale < 1:
            ret_images[i] = pad_around_centre(ret_images[i], w, h, fill)
        elif scale > 1:
            ret_images[i] = crop_to_centre(ret_images[i], w, h)

    return ret_images

def random_flip(images):
    if random.randint(0,1) == 1:
        return [cv2.flip(image, 0) for image in images]
    else:
        return images

def deform1(images, background_value=0):
    # ensure_same_shape(images)

    images = random_flip(images)
    images = random_rotate(images, (-180,180), flags=cv2.INTER_NEAREST, fill=background_value)
    # images = random_scale(images, (0.5,1.5), fill=background_value)

    return images