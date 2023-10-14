import torch
from torchvision.datasets import VisionDataset
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import neuralnet._cvutil as u

class WrappedDataLoader:
    def __init__(self, dl, func):
        """This class wraps a post_processing function `func` around dataloader `dl`, as in https://pytorch.org/tutorials/beginner/nn_tutorial.html"""
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

class DeformedDataset(VisionDataset):
    """This class acts as the dataset loader for both the Chinese Characters dataset and my synthetic Stroke dataset, and can be used for any dataset using grayscale square images for both features and labels.

    The constructor loads all the images in the dataset before training begins
    to improve efficiency. The only operations carried out during train-time (in __getitem__) are just-in-time augmentation using `self.transform`, normalisation to intensities 0-1,
    conversion from numpy array to pytorch tensor and a reshape to return (1,width,height) for both feature and label.

    __init__(Args):
        feature_root (directory string): Folder where *grasycale* feature images are located. Must contain *only* those feature images.
        label_root (directory string): Folder where *grayscale* label images are located. Must contain *only* those label images.
        in_dim (int): Side dimension of square images as they are read from feature_root and label_root
        out_dim (int): Side dimension of images to be resized to in __init__
        crop_amount (int, optional): How much to crop the images (before they are resized). Defaults to 0.
        transform (function, optional): Transformation to be applied identically to each feature & label pair. Must accept image tuple (feature, label) as first argument. Will be applied in a just-in-time manner, i.e. when they are fetched from __getitem__. If None, no transformation will be applied. Defaults to None.
    """
    
    def __init__(self, feature_root, label_root, in_dim, out_dim, crop_amount=0, transform=None):
        super().__init__(None, transform=transform)

        # iterate through feature_root and label_root folders, and fill self.features and self.labels respectively with the cropped & rescaled images in those folders.
        print("Loading dataset into memory...")
        feature_names = [file.name for file in os.scandir(feature_root)]
        label_names = [file.name for file in os.scandir(label_root)]
        self.features = []
        self.labels = []
        for feature_name, label_name in zip(feature_names, label_names):
            feature = u.readim(os.path.join(feature_root, feature_name), cv2.IMREAD_GRAYSCALE)
            label = u.readim(os.path.join(label_root, label_name), cv2.IMREAD_GRAYSCALE)
            assert feature.shape == (in_dim, in_dim) and label.shape == (in_dim, in_dim), f"Image dimensions need to be {in_dim} for both feature and label, but were {feature.shape} for feature and {label.shape} for label."
            
            if crop_amount > 0:
                feature = feature[crop_amount:-crop_amount,crop_amount:-crop_amount]
                label = label[crop_amount:-crop_amount,crop_amount:-crop_amount]

            feature = cv2.resize(feature, (out_dim, out_dim), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (out_dim, out_dim), interpolation=cv2.INTER_NEAREST)

            # option to train on the shape outlines instead
            # feature = cv2.Canny(feature, 100, 200)
            # label = cv2.Canny(label, 100, 200)

            self.features.append(feature)
            self.labels.append(label)
        print("Done loading dataset into memory")

        self.length = len(self.features)
        assert self.length == len(self.labels), "there must be the same amount of labels and features"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        if self.transform is not None:
            feature, label = self.transform((feature, label))

        # normalise to 0-1 intensity, convert from numpy array to pytorch tensor, reshape to (1,width,height) for both feature & label.
        # this means feature and label batches will have shape (batch_size, 1, width, height).
        return torch.from_numpy(feature.astype(np.float32)/255).unsqueeze(0), torch.from_numpy(label.astype(np.float32)/255).unsqueeze(0)
    
class PyramidDataset(VisionDataset):
    """This was to be used with the Multiscale UNet model but I did not get along to testing that model fully.

    Takes the similar arguments to DeformedDataset and behaves the same way; except after the crop, each feature & label is subsampled to all the
    dimensions provided in `scales`. For example, a 1024x1024 image with `scales`=(64,128) would return a feature and label that are each a
    list of two pytorch tensors of shape [(1,64,64),(1,128,128)] from __getitem__.
    """
    
    def __init__(self, feature_root, label_root, scales, in_dim, crop_amount=0, transform=None):
        super().__init__(None, transform=transform)

        scales = sorted(scales) # put scales in ascending order

        print("Loading dataset into memory...")
        feature_names = [file.name for file in os.scandir(feature_root)]
        label_names = [file.name for file in os.scandir(label_root)]
        self.feature_pyramids = []
        self.label_pyramids = []
        for feature_name, label_name in zip(feature_names, label_names):
            feature = u.readim(os.path.join(feature_root, feature_name), cv2.IMREAD_GRAYSCALE)
            label = u.readim(os.path.join(label_root, label_name), cv2.IMREAD_GRAYSCALE)
            assert feature.shape == (in_dim, in_dim) and label.shape == (in_dim, in_dim), f"Image dimensions need to be {in_dim} for both feature and label, but were {feature.shape} for feature and {label.shape} for label."
            
            if crop_amount > 0:
                feature = feature[crop_amount:-crop_amount,crop_amount:-crop_amount]
                label = label[crop_amount:-crop_amount,crop_amount:-crop_amount]

            # feature = cv2.resize(feature, (out_dim, out_dim), interpolation=cv2.INTER_NEAREST)
            # label = cv2.resize(label, (out_dim, out_dim), interpolation=cv2.INTER_NEAREST)

            feature_pyramid = []
            label_pyramid = []
            for scale in scales:
                feature_pyramid.append(cv2.resize(feature, (scale, scale), interpolation=cv2.INTER_NEAREST))
                label_pyramid.append(cv2.resize(label, (scale, scale), interpolation=cv2.INTER_NEAREST))

            # feature = cv2.Canny(feature, 100, 200)
            # label = cv2.Canny(label, 100, 200)

            self.feature_pyramids.append(feature_pyramid)
            self.label_pyramids.append(label_pyramid)
        print("Done loading dataset into memory")

        self.length = len(self.feature_pyramids)
        assert self.length == len(self.label_pyramids), "there must be the same amount of labels and features"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        feature_pyramid = self.feature_pyramids[index]
        label_pyramid = self.label_pyramids[index]

        # apply self.transform. An identical transformation is applied to all images in the feature & label pyramids of a single __getitem__ call.
        if self.transform is not None:
            ret = self.transform(feature_pyramid + label_pyramid)
            feature_pyramid = ret[:len(feature_pyramid)]
            label_pyramid = ret[len(feature_pyramid):]

        feature_pyramid = [torch.from_numpy(f.astype(np.float32)/255).unsqueeze(0) for f in feature_pyramid]
        label_pyramid = [torch.from_numpy(l.astype(np.float32)/255).unsqueeze(0) for l in label_pyramid]

        return feature_pyramid, label_pyramid[-1]
    
## DELETE BELOW HERE

class RandomNoise(VisionDataset):
    def __init__(self):
        super().__init__(None)

        self.noise = torch.from_numpy(u.readim("noise64.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)/255).unsqueeze(0)
        self.blank = torch.from_numpy(np.zeros_like(self.noise).astype(np.float32)).unsqueeze(0)
        plt.hist(self.noise.flatten(), bins=20)
        plt.hist(self.blank.flatten(), bins=20)
        plt.show()
        print(self.noise.shape, self.blank.shape)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        # plt.subplot(121), plt.imshow(self.blank, cmap="gray")
        # plt.subplot(122), plt.imshow(self.noise, cmap="gray")
        # plt.show()
        print(self.noise.shape, self.blank.shape)
        return self.blank, self.noise
    
class MemoriseShape(VisionDataset):
    def __init__(self, feature_dir, label_dir, out_dim):
        super().__init__(None)

        self.feature = u.readim(feature_dir, cv2.IMREAD_GRAYSCALE)
        self.label = u.readim(label_dir, cv2.IMREAD_GRAYSCALE)

        self.feature = cv2.resize(self.feature, (out_dim, out_dim), interpolation=cv2.INTER_NEAREST)
        self.label = cv2.resize(self.label, (out_dim, out_dim), interpolation=cv2.INTER_NEAREST)

        # self.feature = cv2.Canny(self.feature, 100,200)
        # self.label = cv2.Canny(self.label, 100,200)

        plt.subplot(121), plt.imshow(self.feature, cmap="gray")
        plt.subplot(122), plt.imshow(self.label, cmap="gray")
        plt.show()

        self.feature = torch.from_numpy(self.feature.astype(np.float32)/255).unsqueeze(0)
        self.label = torch.from_numpy(self.label.astype(np.float32)/255).unsqueeze(0)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        # plt.subplot(121), plt.imshow(self.blank, cmap="gray")
        # plt.subplot(122), plt.imshow(self.noise, cmap="gray")
        # plt.show()
        # print(self.noise.shape, self.blank.shape)
        return self.feature, self.label