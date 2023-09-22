import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

import dataloader
import model, ex_model

def normalise(img):
    img -= img.min()
    img /= img.max()
    return img

dev = torch.device("cpu")
print("Using CPU.")

ce = torch.nn.BCELoss()

dataset_dir = r"E:\greg\Chinese Characters\3D Printed Deformations\SyntheticStrokeDataset1"
valid_dir = os.path.join(dataset_dir, "Test")

w,h = 128,128
def preprocess(x, y):
    return x.view(-1, 1, w, h).to(dev), y.view(-1, w, h).to(dev)

my_model = ex_model.UNet((w,h), do_output_sigmoid=True, do_batchnorm=False, do_layernorm=True).to(dev)
my_model.load_state_dict(torch.load(r"E:\greg\Chinese Characters\3D Printed Deformations\trained_params\20230922-122935.torchparams"))

valid_ds = dataloader.DeformedDataset(os.path.join(valid_dir, "Features"), os.path.join(valid_dir, "Labels"), (128,128), (w,h), 0)
# valid_ds = dataloader.MemoriseShape((w,h))
valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False)

# put test eval as a check

# preset="f1"
# valid_ds = dataloader.DeformedDataset(r"E:\greg\CharacterDeform\random", r"E:\greg\CharacterDeform\random", (w,h), preset=preset)
# valid_dl =dataloader.WrappedDataLoader(DataLoader(valid_ds, batch_size=1, shuffle=False), preprocess)

alpha = 0.5
with torch.no_grad():
    for xb, yb in valid_dl:
        # img1 = cv2.cvtColor(xb[0][0].numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # overlay = np.zeros((h,w,3), np.uint8)
        # overlay[:,:,1] = yb.numpy()
        # img1 = cv2.addWeighted(img1, alpha, overlay, 1-alpha, 0)

        # img2 = cv2.cvtColor(xb[0][0].numpy().astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # # print(img2.dtype, img2.shape)
        # overlay2 = np.zeros((h,w,3), np.uint8)
        # overlay2[:,:,1] = my_model(xb)[0].numpy()
        # img2 = cv2.addWeighted(img2, alpha, overlay2, 1-alpha, 0)
        # # print(overlay.dtype, overlay.shape)

        # plt.subplot(131), plt.imshow(img1)
        # plt.subplot(132), plt.imshow(img2)

        # plt.subplot(133), plt.imshow(my_model(xb)[0])
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt.show()
        
        print(f"{ce(my_model(xb), yb):.4f}, {torch.nn.functional.mse_loss(my_model(xb), yb)}")
        print(xb.shape, yb.shape, my_model(xb).shape)
        plt.subplot(131), plt.imshow(xb[0][0], cmap="gray")
        plt.title("Feature")
        plt.subplot(132), plt.imshow(yb[0], cmap="gray")
        plt.title("Label")
        plt.subplot(133), plt.imshow(my_model(xb)[0], cmap="gray")
        plt.title("Prediction")
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()