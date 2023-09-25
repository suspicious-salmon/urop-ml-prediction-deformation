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
            for p in model.parameters(): p.grad = None # replaces optimiser.zero_grad()

            if log_standard_loss:
                standard_train_mse_loss += torch.nn.functional.mse_loss(result, yb)

            if show_plot and idx == len(train_dl)-1 and (epoch-1)%10==0:
                display_xb = xb[-1] if type(xb) is list else xb
                plt.subplot(231), plt.imshow(display_xb[-1][0].to(cpu).detach(), cmap="gray")
                plt.title("Train Feature")
                plt.subplot(232), plt.imshow(yb[-1][0].to(cpu).detach(), cmap="gray")
                plt.title("Train Label")
                plt.subplot(233), plt.imshow(result[-1][0].to(cpu).detach(), cmap="gray")
                plt.title("Train Prediction")
                plt.show()
        
        train_losses.append(train_loss / len(train_dl))
        if log_standard_loss:
            standard_train_mse_losses.append(standard_train_mse_loss / len(train_dl))

        # Evaluate against entire validation set
        model.eval()
        valid_loss = 0
        standard_valid_mse_loss = 0
        with torch.no_grad():
            for idx, (xb, yb) in (pbar := tqdm(enumerate(valid_dl), total=len(valid_dl))):

                result = model(xb)

                loss = loss_function(result, yb)
                pbar.set_description(f"VL LOSS: {loss.item():.4f}")
                pbar.refresh()
                valid_loss += loss.item()

                if log_standard_loss:
                    standard_valid_mse_loss += torch.nn.functional.mse_loss(result, yb)

                if show_plot and idx == len(valid_dl)-1 and epoch%10==0:
                    display_xb = xb[-1] if type(xb) is list else xb
                    plt.subplot(234), plt.imshow(display_xb[-1][0].to(cpu).detach(), cmap="gray")
                    plt.title("Test Feature")
                    plt.subplot(235), plt.imshow(yb[-1][0].to(cpu).detach(), cmap="gray")
                    plt.title("Test Label")
                    plt.subplot(236), plt.imshow(result[-1][0].to(cpu).detach(), cmap="gray")
                    plt.title("Test Prediction")
                    # plt.show()

        valid_losses.append(valid_loss / len(valid_dl))
        if log_standard_loss:
            standard_valid_mse_losses.append(standard_valid_mse_loss / len(valid_dl))

        print(f"Epoch: {epoch}, train loss: {train_losses[-1]}, valid loss: {valid_losses[-1]}")

        if log_wandb:
            print("Logging wandb")
            if log_standard_loss:
                print(f"Logging standard: {standard_train_mse_losses[-1]} {standard_valid_mse_losses[-1]}")
                wandb.log({"standard_training_loss": standard_train_mse_losses[-1], "standard_validation_loss": standard_valid_mse_losses[-1],
                           "training_loss": train_losses[-1], "validation_loss": valid_losses[-1]})
            else:
                wandb.log({"training_loss": train_losses[-1], "validation_loss": valid_losses[-1]})