"""This is the master file for running neural network training experiments.
It is written to be run in the terminal, since that is faster than vscode debug mode.
The only argument is whether or not to record training information on Weights&Biases. An example of running it in terminal:

> conda activate <pytorch environment>
> python .\experiment.py True
This will run the experiments with Weights&Biases recording enabled.

This file does one run for every posssible combination of parameters in PARAMS_DICT and saves the trained parameters in folder params_folder."""

import torch
from torch.utils.data import DataLoader
import torchinfo
import wandb
import os
import time
import sys

import _train as _train, _dataloader as _dataloader, _augment as _augment
from models import _unet
from models import _deep_res_unet, _multi_scale

# Use the graphics card for training if one is available; if not use CPU
if torch.cuda.is_available():
     dev = torch.device("cuda")
     print("CUDA available. Using CUDA.")
else:
     dev = torch.device("cpu")
     print("CUDA not available. Using CPU.")

OPTIMISERS = {
     "SGD" : torch.optim.SGD,
     "Adam" : torch.optim.Adam,
}

LOSS_FUNCTIONS = {
     "MSE" : torch.nn.functional.mse_loss,
     "BCE" : torch.nn.BCELoss(),
}

TRANSFORMS = {
    "None" : None,
    "RandomRotateFlip" : _augment.deform1,
    "RandomRotateFlipScale" : _augment.deform2,
}

def run(optimiser_name, optimiser_args, loss_function_name, batch_size, epochs, transform_name, do_batchnorm, do_layernorm, clip_value, log_wandb, wrap_func, pin_memory=True):
     """Runs the training for the given set of parameters. Initialises a new model & model parameters each time this is called."""

     # set train dataset's transform and create dataloaders
     # pin_memory=True improves performance but can cause out of memory exceptions
     train_ds.transform = TRANSFORMS[transform_name]
     train_dl = _dataloader.WrappedDataLoader(DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory), func=wrap_func)
     valid_dl = _dataloader.WrappedDataLoader(DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory), func=wrap_func)

     # Whether you need a sigmoid at the model output depends on if your error function already contains some form of normalisation. BCE and MSE do not, so use do_output_sigmoid should be True for these.
     my_model = _unet.UNet(out_dim, do_output_sigmoid=True, do_layernorm=True, do_batchnorm=do_batchnorm).to(dev)
     # my_model = deep_res_unet.DeepResUnet(out_dim, 6, do_output_sigmoid=True, do_layernorm=do_layernorm, do_batchnorm=do_batchnorm, do_residual=True).to(dev)
     # my_model = multi_scale.MultiScale(model.UNet, scales=(32,128), do_output_sigmoid=True, do_layernorm=True).to(dev)

     try:
          print(torchinfo.summary(my_model.to(torch.device("cpu")), input_size=(batch_size, 1, out_dim, out_dim)))
     except RuntimeError:
          print("Not printing model summary; torchinfo failed (RuntimeError).")
     optimiser = OPTIMISERS[optimiser_name](my_model.parameters(), **optimiser_args)
     loss_function = LOSS_FUNCTIONS[loss_function_name]

     # current time in string form to use in trained parameters filename and to save to Weights&Biases
     start_time = time.strftime('%Y%m%d-%H%M%S')
     # initialise Weights&Biases
     if log_wandb: wandb.init(
          # set the wandb project where this run will be logged
          project="Chinese Characters",

          # track hyperparameters and metadata
          config={
               "architecture": "DeepresUNet with Layernorm",
               "dataset": "Chinese Characters",
               "img_dimensions" : out_dim,
               "optimiser" : optimiser_name,
               **optimiser_args,
               "loss_function" : loss_function_name,
               "batch_size" : batch_size,
               "epochs": epochs,
               "transform" : transform_name,
               "do_batchnorm" : do_batchnorm,
               "do_layernorm" : do_layernorm,
               "clip_value" : "None" if clip_value is None else clip_value,
               "time" : start_time,
          }
     )

     # record gradients & parameters
     if log_wandb: wandb.watch(
          my_model,
          criterion = loss_function,
          log = "all",
          log_freq = 1,
     )

     # train model, save its trained parameters, and finish Weights&Biases
     _train.train(my_model, loss_function, optimiser, train_dl, valid_dl, epochs, batch_size, dev, show_plot=False, log_wandb=log_wandb, clip_value=clip_value, log_standard_loss=True)
     torch.save(my_model.state_dict(), os.path.join(params_folder, f"{start_time}.torchparams"))
     if log_wandb: wandb.finish()

# Tolder to save the parameters of the model after training. They are saved as {time at start of training}.torchparams so there is no fear of overwriting.
# This time is also saved in the Weights&Biases log.
params_folder = r"E:\greg\Chinese Characters\3D Printed Deformations\trained_params"
assert os.path.isdir(params_folder), f"Parameters folder does not exist at {params_folder}"

# Fixed Parameters
optimiser = "Adam"
loss_function = "BCE"
epochs = 40
clip_value = 1 # gradients will be clipped at plusminus clip_value.

# Parameters to try. The model will be trained for all possible combinations of these.
PARAMS_DICT = {
    "learning_rates" : [0.003, 0.0003],
    "betas" : [(0.9, 0.999)],
    "weight_decay" : [0],
    "transforms" : ["RandomRotateFlipScale"],
    "batch_size" : [32, 8],
    "do_batchnorm" : [False],
    "do_layernorm" : [True],
}

# Parameters for the dataset, load the dataset
in_dim = 2460
out_dim = 128

# Load Chinese Character dataset for UNet or DeepResUnet
ds_dir = r"E:\greg\Chinese Characters\3D Printed Deformations\MLDataset"
train_ds = _dataloader.DeformedDataset(os.path.join(ds_dir, "Train", "Features"), os.path.join(ds_dir, "Train", "Labels"), in_dim, out_dim)
valid_ds = _dataloader.DeformedDataset(os.path.join(ds_dir, "Test", "Features"), os.path.join(ds_dir, "Test", "Labels"), in_dim, out_dim)

# Load Chinese Character dataset for multi-scale nn
# scales = (32, 128)
# ds_dir = r"E:\greg\Chinese Characters\3D Printed Deformations\MLDataset"
# train_ds = dataloader.PyramidDataset(os.path.join(ds_dir, "Train", "Features"), os.path.join(ds_dir, "Train", "Labels"), scales, in_dim)
# valid_ds = dataloader.PyramidDataset(os.path.join(ds_dir, "Test", "Features"), os.path.join(ds_dir, "Test", "Labels"), scales, in_dim)

# Wrapper functions to send the batch tensors to the right device (gpu or cpu). Use wrap_func for all but the Multiscale UNet. For the Multiscale UNet, use wrap_func_pyramid.
def wrap_func_pyramid(x, y):
     return [i.to(dev) for i in x], [i.to(dev) for i in y]
def wrap_func(x, y):
     return x.to(dev), y.to(dev)

def main(log_wandb):
     log_wandb = log_wandb.lower() == "true"
     print(f"Log_wandb: {log_wandb}")

     for transform in PARAMS_DICT["transforms"]:
          for betas in PARAMS_DICT["betas"]:
               for do_batchnorm in PARAMS_DICT["do_batchnorm"]:
                    for do_layernorm in PARAMS_DICT["do_layernorm"]:
                         for batch_size in PARAMS_DICT["batch_size"]:
                              for weight_decay in PARAMS_DICT["weight_decay"]:
                                   for learning_rate in PARAMS_DICT["learning_rates"]:      

                                        args = (optimiser,
                                             {"learning_rate" : learning_rate,
                                             "betas" : betas,
                                             "weight_decay" : weight_decay,
                                             "amsgrad" : True}, # ^ this dictionary has the optimiser arguments
                                             loss_function, 
                                             batch_size,
                                             epochs,
                                             transform,
                                             do_batchnorm,
                                             do_layernorm,
                                             clip_value,
                                             log_wandb,
                                             wrap_func,
                                             True)
                                        print(args)
                                        run(*args)
                    
if __name__ == "__main__":
    terminal_args = sys.argv[1:]
    main(*terminal_args)