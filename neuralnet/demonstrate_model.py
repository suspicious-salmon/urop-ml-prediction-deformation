import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import cv2

import _dataloader
import models._unet

# Wrapper functions to send the batch tensors to the right device (gpu or cpu). Use wrap_func for all but the Multiscale UNet. For the Multiscale UNet, use wrap_func_pyramid.
def wrap_func_pyramid(x, y):
     return [i.to(dev) for i in x], [i.to(dev) for i in y]
def wrap_func(x, y):
     return x.to(dev), y.to(dev)

dev = torch.device("cpu")
print("Using CPU.")

# load Chinese Characters test set
dataset_dir = r"E:\UROP\Machine Learning\Datasets\ChineseCharacterDataset"
valid_dir = os.path.join(dataset_dir, "Test")
valid_ds = _dataloader.DeformedDataset(os.path.join(dataset_dir, "Test", "Features"), os.path.join(dataset_dir, "Test", "Labels"), 2460, 128)
valid_dl = _dataloader.WrappedDataLoader(DataLoader(valid_ds, batch_size=1, shuffle=False), func=wrap_func)

# set model
my_model = models._unet.UNet(128, do_layernorm=True).to(dev)
my_model.load_state_dict(torch.load(r"E:\UROP\Machine Learning\trained_parameters\worthy-glade.torchparams"))

out_dir = r"C:\Users\gregk\Documents\MyDocuments\Brogramming\Python\UROP\urop-structured-nn\precorrect"

with torch.no_grad():
     for idx, (xb, yb) in enumerate(valid_dl):
          plt.subplot(241), plt.imshow(xb[0][0], cmap="gray")
          plt.title("Feature (deformed 3d-print)")
          plt.subplot(242), plt.imshow(my_model(xb)[0][0], cmap="gray")
          plt.title("Predicted CAD image")
          plt.subplot(243), plt.imshow(cv2.threshold(my_model(xb)[0][0].detach().numpy()*255, 100, 255, cv2.THRESH_BINARY)[1], cmap="gray")
          plt.title("Prediction, thresholded")
          plt.subplot(244), plt.imshow(yb[0][0], cmap="gray")
          plt.title("Label (CAD image)")
          img = my_model(yb)[0][0].detach().numpy()*255
          precorrect = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
          plt.subplot(245), plt.imshow(precorrect, cmap="gray")
          plt.title("Precorrected")
          plt.show()

          cv2.imwrite(os.path.join(out_dir, f"{idx}.tif"), precorrect)

          # figManager = plt.get_current_fig_manager()
          # figManager.window.showMaximized()
          # plt.show()