import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import dataloader
import augment

fd = r"C:\Users\gregk\Documents\MyDocuments\Brogramming & Electronics\Python\multiscale nn\Data\Features"
ld = r"C:\Users\gregk\Documents\MyDocuments\Brogramming & Electronics\Python\multiscale nn\Data\Labels"

scales = (1024, 64)
train_ds = dataloader.PyramidDataset(fd, ld, scales, (1024,1024), (1024,1024), transform=augment.deform1)

train_dl = DataLoader(train_ds, batch_size=1, shuffle=False, pin_memory=True)

for x,y in train_dl:
    for idx, i in enumerate(x):
        plt.subplot(131+idx), plt.imshow(i[0], cmap="gray")
    plt.show()

    for idx, i in enumerate(y):
        plt.subplot(131+idx), plt.imshow(i[0], cmap="gray")
    plt.show()