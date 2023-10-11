"""This module contains the function that trains a model. It can record results on Weights&Biases."""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

def normalise(img):
    img -= img.min()
    img /= img.max()
    return img

cpu = torch.device("cpu")

def train(model: torch.nn.Module, loss_function: function, optimiser: torch.optim.optimizer, train_dl: DataLoader, valid_dl: DataLoader, epochs: int, batch_size: int, dev: torch.device,
          show_plot=False, log_wandb=True, log_standard_loss=False, clip_value=None):
    """Train model for given number of epochs.

    Args:
        - `model`: pytorch model to train
        - `loss_function`: pytorch loss function to use in training
        - `optimiser`: pytorch optimizer to use in training
        - `train_dl`: training set pytorch dataloader
        - `valid_dl`: validation set pytorch dataloader
        - `epochs`: number of epochs to train for
        - `batch_size`: batch size in all epochs
        - `dev`: torch device to use. e.g. torch.device("cpu") or torch.device("cuda")
        - `show_plot`: whether to display examples of the feature, label and model output during training. (note: this interrupts training until the matplotlib window is closed.)
        - `log_wandb`: whether to save training & validation losses to Weights&Biases
        - `log_standard_loss`: whether to also save mean square error (MSE) losses to Weights&Biases
        - `clip_value`: value to clip all gradients at. e.g. if clip_value is 1, gradients with magnitude greater than 1 will be clipped to magnitude 1.
    """

    train_losses = []
    valid_losses = []
    standard_train_mse_losses = []
    standard_valid_mse_losses = []

    for epoch in tqdm(range(epochs)):

        # Train
        # --------------------

        model.train()
        epoch_train_loss = 0
        standard_train_mse_loss = 0
        for idx, (xb, yb) in (pbar := tqdm(enumerate(train_dl), total=len(train_dl))):
            result = model(xb)

            # calculate average training loss for this batch
            loss = loss_function(result, yb)
            epoch_train_loss += loss.item()

            # add training loss to progress bar
            pbar.set_description(f"TR LOSS: {loss.item():.4f}")
            pbar.refresh()

            # calculate gradients
            loss.backward()
            # clip gradients
            if clip_value is not None:
                for p in model.parameters():
                    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
            # apply gradient descent to parameters
            optimiser.step()
            for p in model.parameters(): p.grad = None # this line replaces optimiser.zero_grad(), is slightly more efficient

            # calculate average training mean square error (MSE) loss for this batch as well. different training runs might try different loss functions - calculating MSE loss for all of them allows comparison of performance.
            if log_standard_loss:
                standard_train_mse_loss += torch.nn.functional.mse_loss(result, yb)

            # can be used to view the output of the model at this epoch. Displays the feature, label and model output for that feature.
            if show_plot and idx == len(train_dl)-1 and (epoch-1)%10==0:
                display_xb = xb[-1] if type(xb) is list else xb
                plt.subplot(231), plt.imshow(display_xb[-1][0].to(cpu).detach(), cmap="gray")
                plt.title("Train Feature")
                plt.subplot(232), plt.imshow(yb[-1][0].to(cpu).detach(), cmap="gray")
                plt.title("Train Label")
                plt.subplot(233), plt.imshow(result[-1][0].to(cpu).detach(), cmap="gray")
                plt.title("Train Prediction")
                plt.show()
        
        # calculate average training loss for this epoch
        train_losses.append(epoch_train_loss / len(train_dl))
        if log_standard_loss:
            # calculate average training MSE loss for this epoch
            standard_train_mse_losses.append(standard_train_mse_loss / len(train_dl))

        # Validation
        # -----------

        # Evaluate against entire validation set
        model.eval()
        epoch_valid_loss = 0
        standard_valid_mse_loss = 0
        with torch.no_grad():
            for idx, (xb, yb) in (pbar := tqdm(enumerate(valid_dl), total=len(valid_dl))):
                result = model(xb)

                # calculate average validation loss for this batch
                loss = loss_function(result, yb)
                epoch_valid_loss += loss.item()

                # add validation loss to progress bar
                pbar.set_description(f"VL LOSS: {loss.item():.4f}")
                pbar.refresh()

                # calculate average validation MSE loss for this batch
                if log_standard_loss:
                    standard_valid_mse_loss += torch.nn.functional.mse_loss(result, yb)

                # can be used to view the output of the model on the validation set at this epoch. Displays the feature, label and model output for that feature.
                if show_plot and idx == len(valid_dl)-1 and epoch%10==0:
                    display_xb = xb[-1] if type(xb) is list else xb
                    plt.subplot(234), plt.imshow(display_xb[-1][0].to(cpu).detach(), cmap="gray")
                    plt.title("Test Feature")
                    plt.subplot(235), plt.imshow(yb[-1][0].to(cpu).detach(), cmap="gray")
                    plt.title("Test Label")
                    plt.subplot(236), plt.imshow(result[-1][0].to(cpu).detach(), cmap="gray")
                    plt.title("Test Prediction")

        # calculate average validation loss for this epoch
        valid_losses.append(epoch_valid_loss / len(valid_dl))
        # print average training and validation loss from this epoch
        print(f"Epoch: {epoch}, train loss: {train_losses[-1]}, valid loss: {valid_losses[-1]}")
        if log_standard_loss:
            # calculate average validation MSE loss for this epoch
            standard_valid_mse_losses.append(standard_valid_mse_loss / len(valid_dl))
            # print average training and validaiton MSE loss for this epoch
            print(f"mse train loss: {standard_train_mse_losses[-1]}, mse epoch_valid_loss: {standard_valid_mse_losses[-1]}")

        # save losses to Weights&Biases
        if log_wandb:
            print("Logging wandb")
            if log_standard_loss:
                print(f"Logging standard: {standard_train_mse_losses[-1]} {standard_valid_mse_losses[-1]}")
                wandb.log({"standard_training_loss": standard_train_mse_losses[-1], "standard_validation_loss": standard_valid_mse_losses[-1],
                           "training_loss": train_losses[-1], "validation_loss": valid_losses[-1]})
            else:
                wandb.log({"training_loss": train_losses[-1], "validation_loss": valid_losses[-1]})