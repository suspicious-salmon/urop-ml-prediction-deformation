import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import scipy
import cv2
import os
from tqdm import tqdm
from pathlib import Path

import utility as u

# for high level features: shrink by mapping to r,theta then smoothing
# for low level features: make edges change from zigzag to sine

def shrink1(x, a=0.9, b=10):
        norm = x - x.mean()
        shrunk = b * np.log(np.abs(norm)/b + 1)
        return a * (np.sign(norm) * shrunk + x.mean())

class Rect:
    def __init__(self, topleft, bottomright):
        self.topleft = topleft
        self.bottomright = bottomright

        self.de_parametrise = np.vectorize(self.de_parametrise)

    @property
    def width(self):
        return self.bottomright[0] - self.topleft[0]
    @property
    def height(self):
        return self.topleft[1] - self.bottomright[1]

    @property
    def area(self):
        return self.width * self.height
    @property
    def centroid(self):
        return (self.topleft[0]+self.bottomright[0])/2, (self.topleft[1]+self.bottomright[1])/2
    
    def parametrise(self):
        # theta values for topleft and bottomright corner
        centroid = self.centroid
        topleft_angle = math.pi+math.atan((self.topleft[1]-centroid[1])/(self.topleft[0]-centroid[0]))
        bottomright_angle = math.atan((self.bottomright[1]-centroid[1])/(self.bottomright[0]-centroid[0]))
        
        def r(t):
            theta = t % math.pi
            if abs(theta) <= topleft_angle and abs(theta) >= -bottomright_angle:
                return abs(self.height/2 / math.sin(theta)) + np.sin(50*theta)
            else:
                return abs(self.width/2 / math.cos(theta)) + np.sin(50*theta)
            
        return np.vectorize(r)
    
    def de_parametrise(self, r, theta):
        x = r * np.cos(theta) + self.centroid[0]
        y = r * np.sin(theta) + self.centroid[1]
        return x, y
    
    def blit_mpl(self):
        return patches.Rectangle(self.topleft, self.width, -self.height, linewidth=1, edgecolor='k', facecolor='none')
    
class Circle:
    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius

        self.de_parametrise = np.vectorize(self.de_parametrise)

    @property
    def area(self):
        return np.pi * self.radius*self.radius
    @property
    def centroid(self):
        return self.centre
    
    def parametrise(self):
        return np.vectorize(lambda t : self.radius)
    
    def blit_mpl(self):
        return patches.Circle(self.centre, self.radius, linewidth=1, edgecolor="k", facecolor="none")
    
    def de_parametrise(self, r, theta):
        x = r * np.cos(theta) + self.centroid[0]
        y = r * np.sin(theta) + self.centroid[1]
        return x, y
    
class Stroke:
    def __init__(self):
        # leave enough space for random rotation of the curve without cutting any off
        low = 0.5*(1 - 1/np.sqrt(2))
        high = 0.5*(1 + 1/np.sqrt(2))
        self.root_points_x = np.random.uniform(low, high, size=(3))
        self.root_points_y = np.random.uniform(low, high, size=(3))

        # plt.scatter(self.root_points_x, self.root_points_y), plt.xlim(0,1), plt.ylim(0,1), plt.show()
        
        self.arc_fn_x = scipy.interpolate.CubicSpline(np.array([0,1,2]), self.root_points_x)
        self.arc_fn_y = scipy.interpolate.CubicSpline(np.array([0,1,2]), self.root_points_y)

        N=1000
        shorten_factor = 0.3 # how much the length of the line should shorten by (will be split between both ends)
        self.arc_points_in_x = self.arc_fn_x(np.linspace(0,2,N))
        self.arc_points_in_y = self.arc_fn_y(np.linspace(0,2,N))
        self.arc_points_out_x = self.arc_fn_x(np.linspace(shorten_factor,2-shorten_factor,N))
        self.arc_points_out_y = self.arc_fn_y(np.linspace(shorten_factor,2-shorten_factor,N))

        image_dim = 128
        self.image = np.zeros((image_dim, image_dim), np.uint8)
        self.shrunk = np.zeros((image_dim, image_dim), np.uint8)
        self.image[np.floor(self.arc_points_in_x*image_dim).astype(np.uint8), np.floor(self.arc_points_in_y*image_dim).astype(np.uint8)] = 255
        self.shrunk[np.floor(self.arc_points_out_x*image_dim).astype(np.uint8), np.floor(self.arc_points_out_y*image_dim).astype(np.uint8)] = 255

        in_dilate_size = int(0.03*image_dim)
        out_dilate_size = int(0.06*image_dim)
        self.image_dilated = cv2.dilate(self.image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (in_dilate_size, in_dilate_size)))
        self.shrunk_dilated = cv2.dilate(self.shrunk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (out_dilate_size, out_dilate_size)))

        # plt.subplot(121), plt.imshow(self.image_dilated, cmap="gray")
        # plt.subplot(122), plt.imshow(self.shrunk_dilated, cmap="gray")
        # plt.show()

class Scene:
    def __init__(self, shapes=None):
        if shapes is None:
            self.shapes = []
        else:
            self.shapes = shapes
            self._centroid = self.centroid # centroid at initialisation, can be used to save time if scene is not modified

    @property
    def centroid(self):
        assert len(self.shapes) > 0, "cannot find centroid of Scene, shapes array is empty"
        x = sum([shape.centroid[0] * shape.area for shape in self.shapes]) / sum([shape.area for shape in self.shapes])
        y = sum([shape.centroid[1] * shape.area for shape in self.shapes]) / sum([shape.area for shape in self.shapes])
        return x, y
    
    def output_scene(self, ax):
        for shape in self.shapes:
            ax.add_patch(shape.blit_mpl())
        # ax.autoscale()
        # plt.show()
    
    def output_shrunk_scene(self, a=0.9, b=10, resolution=1000):
        t = np.arange(0, 2*np.pi, 1/resolution)
        shape_coords = []
        for shape in self.shapes:
            r = shrink(shape.parametrise()(t), a, b)
            print(shape.de_parametrise(r, t))
            shape_coords.append(shape.de_parametrise(r, t))
        return shape_coords

# for i in range(10): Stroke()

dataset_dir = r"E:\greg\Chinese Characters\3D Printed Deformations\SyntheticStrokeDataset1"
assert os.path.isdir(dataset_dir)
Path(os.path.join(dataset_dir, "Test", "Features")).mkdir(parents=True, exist_ok=False)
Path(os.path.join(dataset_dir, "Test", "Labels")).mkdir(parents=True, exist_ok=False)
Path(os.path.join(dataset_dir, "Train", "Features")).mkdir(parents=True, exist_ok=False)
Path(os.path.join(dataset_dir, "Train", "Labels")).mkdir(parents=True, exist_ok=False)

N_examples = 1000
N_test = 200
# randomly select a number N_test indexes to be the test set
index_shuffle = np.random.default_rng().permuted(np.arange(0, N_examples, 1).astype(int))
test_idxs = index_shuffle[:N_test]

# make synthetic dataset
for i in tqdm(range(N_examples)):
    stroke = Stroke()
    example_type = "Test" if i in test_idxs else "Train"
    u.writeim(os.path.join(dataset_dir, example_type, "Features", f"{i}_stroke.png"), stroke.shrunk_dilated)
    u.writeim(os.path.join(dataset_dir, example_type, "Labels", f"{i}_stroke.png"), stroke.image_dilated)

# rect1 = Rect((-100,100), (0,50))
# circle1 = Circle((-50, 75), 5)
# rect1 = Rect((-25,25), (25,-25))
# circle1 = Circle((0,0), 40)
# shape1 = Scene((rect1, circle1))

# rect1 = Rect((-25,25), (25,-25))
# circle1 = Circle((0,0), 50)
# shape1 = Scene((rect1, circle1))

# VIEW_SCALE=5
# d = 5 # in px
# max_x, min_x, max_y, min_y = 50+d, -50-d, 50+d, -50-d
# width = max_x - min_x
# height = max_y - min_y
# aspect_ratio = width / height
# fig = plt.figure(frameon = False)
# fig.set_size_inches(VIEW_SCALE, VIEW_SCALE)
# axis = plt.Axes(fig, [0.,0.,1.,1.])
# axis.set_axis_off()
# centroid_x, centroid_y = shape1.centroid
# lim_rad = max(max_x - centroid_x, max_y - centroid_y, centroid_x - min_x, centroid_y - min_y)
# axis.set_xlim(centroid_x - lim_rad, centroid_x + lim_rad)
# axis.set_ylim(centroid_y - lim_rad, centroid_y + lim_rad)
# fig.add_axes(axis)

# for shape in shape1.shapes:
#     plt.gca().add_patch(shape.blit_mpl())
# for coords in shape1.output_shrunk_scene():
#     plt.plot(*coords, c="k", linewidth=1)
# shape1.output_scene(axis)

# plt.show()
# fig.savefig("output2.png", dpi=1024/VIEW_SCALE)