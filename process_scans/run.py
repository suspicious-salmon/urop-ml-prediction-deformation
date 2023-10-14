"""This was used to run the data preparation pipeline for the chinese characters dataset.
It takes the nominal source images and ct scan images and outputs 2460x2460 aligned results."""

import os
from tqdm import tqdm
from pathlib import Path
import cv2

import _scaleupfont
import _metrics
import _heatmap
import _align
import _cvutil

output_folder = r"E:\greg\CharacterDeform\Results\Run2"
scans_folder = r"E:\greg\CharacterDeform\1st batch\myruns\my_actual_unaligned"
font_folder = r"E:\greg\CharacterDeform\1st batch\nominal_fonts"

# create results folder structure
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "cads")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "scans")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "aligned_scans")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "extra_pixels")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "missing_pixels")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "heatmaps")).mkdir(parents=True, exist_ok=True)

N=144
for idx in tqdm(range(N)):
    try:
        # copy scan file and invert
        scan_img = _cvutil.readim(os.path.join(scans_folder, f"{idx+1}.tif"), cv2.IMREAD_GRAYSCALE)
        scan_img = 255 - scan_img
        _cvutil.writeim(os.path.join(output_folder, "scans", f"{idx+1}_scan.tif"), scan_img)

        # scale up cad to same scale as scan
        _scaleupfont.scaleup(os.path.join(font_folder, f"{idx+1}.tif"),
                            os.path.join(output_folder, "cads", f"{idx+1}_cad.tif"),
                            2460, 2460)

        # align scan to cad
        _align.align_ccorr(
            os.path.join(output_folder, "cads", f"{idx+1}_cad.tif"),
            os.path.join(output_folder, "scans", f"{idx+1}_scan.tif"),
            {
                "aligned" : os.path.join(output_folder, "aligned_scans", f"{idx+1}_aligned.tif"),
                "extra_pixels" : os.path.join(output_folder, "extra_pixels", f"{idx+1}_extra.tif"),
                "missing_pixels" : os.path.join(output_folder, "missing_pixels", f"{idx+1}_missing.tif")
            },
            angle_bounds=(-30,30)
        )
        
        # # processing introduces some artifacts around previous crop. remove these by cropping again with a slightly smaller kernel.
        # align.crop_to_inflated_cad(os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
        #                            os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
        #                            os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
        #                            kernel_size=275,
        #                            overwrite=True)

        # heatmap
        _heatmap.heatmap(os.path.join(output_folder, "cads", f"{idx+1}_cad.tif"),
                        os.path.join(output_folder, "aligned_scans", f"{idx+1}_aligned.tif"),
                        os.path.join(output_folder, "heatmaps", f"{idx+1}_heatmap.tif"))
        
    except OSError as e:
        print(f"Skipped {idx} due to OSError. Error was: {e}")
        
_metrics.get_directory_metrics(os.path.join(output_folder, "heatmaps"), output_folder)