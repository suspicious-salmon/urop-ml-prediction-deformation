import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import _cvutil

def _get_img_metrics(img):
    """Returns proportion of pixels in img that are red (i.e. lost during printing) and that are blue (i.e. gained during printing), and the sum of those proportions.

    Args:
        img (numpy array): opencv image. heatmap to analyse.

    Returns:
        tuple: (proportion red (lost) pixels, proportion blue (gained) pixels, sum of previous two values (total changed pixels))
    """
    
    img_metric = img.copy()
    # make all white pixels black (to make only red and blue left)
    img_metric[cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) == 255] = (0,0,0)
    # red channel lost material
    lost_material = (img_metric[:,:,0] / 255.).astype(int)
    # blue channel gained material
    gained_material = (img_metric[:,:,2] / 255.).astype(int)

    pixelcount = img_metric.shape[0] * img_metric.shape[1]
    prop_lost_px = lost_material.sum() / pixelcount
    prop_gained_px = gained_material.sum() / pixelcount
    prop_changed_px = (prop_lost_px + prop_gained_px)

    return (prop_lost_px, prop_gained_px, prop_changed_px)

def get_directory_metrics(heatmap_folder, dest_folder):
    """Generates metrics from a folder containing heatmap images generated with _heatmap.py, and saves them as a metrics.json in dest_folder (unless a metrics.json already exists, in which case will save to metrics_overwrite.json).
    It measures:
    - prop_lost_px: proportion (0 to 1) of red pixels in the heatmap image
    - prop_gained_px: propotion of blue pixels in the heatmap image
    - prop_changed_px: sum of prop_lost_px and prop_gained_px

    Args:
        heatmap_folder (directory string): folder to read heatmaps from. the folder must contain only heatmap images.
        dest_folder (directory string): folder (not filename) in which to save metrics.json
    """

    # make metrics dataframe
    df_list = []
    for file in tqdm(os.scandir(heatmap_folder), total=len(list(os.scandir(heatmap_folder)))):
        heatmap = cv2.cvtColor(_cvutil.readim(os.path.join(heatmap_folder, file.name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        df_list.append((file.name.split("_")[0], *_get_img_metrics(heatmap)))

    # save dataframe
    df = pd.DataFrame(df_list, columns=["serial", "prop_lost_px", "prop_gained_px", "prop_changed_px"])
    if not os.path.isfile(os.path.join(dest_folder, "metrics.json")):
        df.to_json(os.path.join(dest_folder, "metrics.json"))
    else:
        print("WARNING: a metrics file already exists. Metrics will be saved in metrics_overwrite.json. It is at risk of being overwritten next time this code is run.")
        df.to_json(os.path.join(dest_folder, "metrics_overwrite.json"))