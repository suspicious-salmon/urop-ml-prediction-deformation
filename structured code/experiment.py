import torch
from torch.utils.data import DataLoader
import torchinfo
import wandb
import os
import time
import sys

import train, dataloader, augment
import model

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
     "cross_entropy" : torch.nn.functional.cross_entropy
}

TRANSFORMS = {
    "None" : None,
    "RandomRotateFlip" : augment.deform1,
}

def run(optimiser_name, optimiser_args, loss_function_name, batch_size, epochs, transform_name, do_batchnorm, do_layernorm, clip_value, log_wandb, log_period, wrap_func, is_nonstandard_model=False):
     train_ds.transform = TRANSFORMS[transform_name]
     train_dl = dataloader.WrappedDataLoader(DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=not(is_nonstandard_model)), func=wrap_func)
     valid_dl = dataloader.WrappedDataLoader(DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=not(is_nonstandard_model)), func=wrap_func)

     # if not is_nonstandard_model: print(torchinfo.summary(my_model.to(torch.device("cpu")), input_size=(batch_size, 1, out_dim, out_dim)))
     optimiser = OPTIMISERS[optimiser_name](my_model.parameters(), **optimiser_args)
     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor = 0.9, patience = 3)
     loss_function = LOSS_FUNCTIONS[loss_function_name]

     start_time = time.strftime('%Y%m%d-%H%M%S')
     if log_wandb: wandb.init(
          # set the wandb project where this run will be logged
          project="Strokes 1",

          # track hyperparameters and metadata
          config={
               "architecture": "UNet with layernorm",
               "dataset": "Strokes 1000",
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

     if log_wandb: wandb.watch(
          my_model,
          criterion = loss_function,
          log = "all",
          log_freq = 1,
     )

     train.train(my_model, loss_function, optimiser, train_dl, valid_dl, epochs, batch_size, dev, show_plot=True, log_wandb=log_wandb, clip_value=clip_value, log_period=log_period, log_standard_loss=True)
     torch.save(my_model.state_dict(), os.path.join(params_dir, f"{start_time}.torchparams"))
     if log_wandb: wandb.finish()

params_dir = r"E:\greg\Chinese Characters\3D Printed Deformations\trained_params"
assert os.path.isdir(params_dir), f"Parameters folder does not exist at {params_dir}"
dataset_dir = r"E:\greg\Chinese Characters\3D Printed Deformations\SyntheticStrokeDataset1"

# Fixed Parameters
optimiser = "Adam"
loss_function = "BCE"
epochs = 40+2 # +2 allows train function to show final result

PARAMS_DICT = {
    "learning_rates" : [0.0003],
    "betas" : [(0.9, 0.999)],
    "weight_decay" : [0],
    "transforms" : ["None"],
    "batch_size" : [1],
    "do_batchnorm" : [False],
    "do_layernorm" : [True],
}

k = 128
in_dim = 1024
out_dim = k
crop_amount = 0

# train_ds = dataloader.DeformedDataset(os.path.join(dataset_dir, "Train", "Features"), os.path.join(dataset_dir, "Train", "Labels"),
#                                       in_dim, out_dim)
# valid_ds = dataloader.DeformedDataset(os.path.join(dataset_dir, "Test", "Features"), os.path.join(dataset_dir, "Test", "Labels"),
#                                       in_dim, out_dim)

# scales = (64, 32)
train_feature_dir = r"C:\Users\gregk\Documents\MyDocuments\Brogramming & Electronics\Python\urop-structured-nn\Data\Features"
train_label_dir = r"C:\Users\gregk\Documents\MyDocuments\Brogramming & Electronics\Python\urop-structured-nn\Data\Labels"
# train_ds = dataloader.PyramidDataset(train_feature_dir, train_feature_dir, scales, in_dim, crop_amount)
# valid_ds = dataloader.PyramidDataset(train_feature_dir, train_feature_dir, scales, in_dim, crop_amount)
onedir=r"E:\greg\Chinese Characters\3D Printed Deformations\urop-structured-nn\Data\Features\bc.tif"
train_ds = dataloader.MemoriseShape(onedir, onedir, out_dim)
valid_ds = dataloader.MemoriseShape(onedir, onedir, out_dim)

def wrap_func_pyramid(x, y):
     return [i.to(dev) for i in x], [i.to(dev) for i in y]
def wrap_func(x, y):
     return x.to(dev), y.to(dev)

my_model = model.DeepResUnet(out_dim, 6, do_output_sigmoid=True, do_layernorm=True, do_residual=True).to(dev)
# my_model = model.UNet(out_dim, do_output_sigmoid=True, do_layernorm=True).to(dev)
# my_model = model.MultiScale(model.UNet, scales, do_output_sigmoid=True, do_layernorm=True).to(dev)
is_nonstandard_model = False

def main(log_wandb):
     log_wandb = log_wandb.lower() == "true"
     print(f"Log_wandb: {log_wandb}")

     count = 0
     for tr in PARAMS_DICT["transforms"]:
          for bt in PARAMS_DICT["betas"]:
               for db in PARAMS_DICT["do_batchnorm"]:
                    for dl in PARAMS_DICT["do_layernorm"]:
                         for bs in PARAMS_DICT["batch_size"]:
                              for wd in PARAMS_DICT["weight_decay"]:
                                   for lr in PARAMS_DICT["learning_rates"]:      
                                        count += 1

                                        args = (optimiser,
                                             {"lr" : lr,
                                             "betas" : bt,
                                             "weight_decay" : wd,
                                             "amsgrad" : True},
                                             loss_function,
                                             bs,
                                             epochs,
                                             tr,
                                             db,
                                             dl,
                                             1, # clip value
                                             log_wandb,
                                             log_every_n_epochs,
                                             wrap_func,
                                             is_nonstandard_model)
                                        print(args)
                                        run(*args)
                    
if __name__ == "__main__":
    terminal_args = sys.argv[1:]
    log_every_n_epochs = 1
    main(*terminal_args)