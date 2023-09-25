import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np
import cv2

from tqdm import tqdm

def normalise(img):
    img -= img.min()
    img /= img.max()
    return img

cpu = torch.device("cpu")

def train(model, loss_function, optimiser, train_dl, valid_dl, epochs, batch_size, dev,
          show_plot=False, log_wandb=True, log_standard_loss=False, clip_value=None, log_period=1):
    train_losses = []
    valid_losses = []
    standard_train_mse_losses = []
    standard_valid_mse_losses = []

    for epoch in tqdm(range(epochs)):
        # Train
        model.train()
        train_loss = 0
        standard_train_mse_loss = 0
        for idx, (xb, yb) in (pbar := tqdm(enumerate(train_dl), total=len(train_dl))):
            xb, yb = xb.to(dev), yb.to(dev)

            result = model(xb)

            loss = loss_function(result, yb)
            pbar.set_description(f"TR LOSS: {loss.item():.4f}")
            pbar.refresh()
            train_loss += loss.item() # !! .item() MAY BE A BOTTLENECK. copies memory from gpu to ram.

            loss.backward()
            if clip_value is not None:
                for p in model.parameters():
                    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
            optimiser.step()
            for p in model.parameters(): p.grad = None
            # optimiser.zero_grad()

            if log_standard_loss:
                # standard_train_mse_loss += get_standard_mse_loss(result, yb, epoch)
                standard_train_mse_loss += torch.nn.functional.mse_loss(result, yb)

            # if idx == len(train_dl)-1:
            #     plt.subplot(231), plt.imshow(xb[-1][0].to(cpu).detach(), cmap="gray")
            #     plt.title("Feature")
            #     plt.subplot(232), plt.imshow(yb[-1].to(cpu).detach(), cmap="gray")
            #     plt.title("Label")
            #     plt.subplot(233), plt.imshow(result[-1].to(cpu).detach(), cmap="gray")
            #     plt.title("Prediction")
            #     plt.show()
        
        train_losses.append(train_loss / len(train_dl))
        if log_standard_loss:
            standard_train_mse_losses.append(standard_train_mse_loss / len(train_dl))

        # Evaluate against entire validation set
        model.eval()
        valid_loss = 0
        standard_valid_mse_loss = 0
        with torch.no_grad():
            for idx, (xb, yb) in (pbar := tqdm(enumerate(valid_dl), total=len(valid_dl))):
                xb, yb = xb.to(dev), yb.to(dev)

                result = model(xb)

                # if idx == len(valid_dl)-1:
                #     plt.subplot(234), plt.imshow(xb[-1][0].to(cpu).detach(), cmap="gray")
                #     plt.title("Feature")
                #     plt.subplot(235), plt.imshow(yb[-1].to(cpu).detach(), cmap="gray")
                #     plt.title("Label")
                #     plt.subplot(236), plt.imshow(result[-1].to(cpu).detach(), cmap="gray")
                #     plt.title("Prediction")
                #     plt.show()

                loss = loss_function(result, yb)
                pbar.set_description(f"VL LOSS: {loss.item():.4f}")
                pbar.refresh()
                valid_loss += loss.item()

                if log_standard_loss:
                    # standard_valid_mse_loss += get_standard_mse_loss(result, yb, epoch)
                    standard_valid_mse_loss += torch.nn.functional.mse_loss(result, yb)

        valid_losses.append(valid_loss / len(valid_dl))
        if log_standard_loss:
            standard_valid_mse_losses.append(standard_valid_mse_loss / len(valid_dl))

        print(f"Epoch: {epoch}, train loss: {train_losses[-1]}, valid loss: {valid_losses[-1]}")

        if show_plot:
            plt.plot(train_losses, c="k", label="Training loss")
            plt.plot(valid_losses, c="c", label="Validation loss")
            if len(train_losses)==1:
                plt.legend()
                plt.grid()
            plt.pause(0.05)

        if log_wandb:
            print("Logging wandb")
            if log_standard_loss:
                print(f"Logging standard: {standard_train_mse_losses[-1]} {standard_valid_mse_losses[-1]}")
                wandb.log({"standard_training_loss": standard_train_mse_losses[-1], "standard_validation_loss": standard_valid_mse_losses[-1],
                           "training_loss": train_losses[-1], "validation_loss": valid_losses[-1]})
            else:
                wandb.log({"training_loss": train_losses[-1], "validation_loss": valid_losses[-1]})

    plt.show()
