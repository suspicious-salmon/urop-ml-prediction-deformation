# urop-ml-prediction-deformation

## Description

This repository contains the code I used to build the pre-correction model for Sara's Chinese Character dataset. There are three folders:

#### process_scans

This folder contains code to process the CT scans taken of the 3D-printed Chinese Characters.

Before running this, I segmented the scans in VGStudio and manually cropped each individual chinese character into its own image (although modifying crop.py, written by Sara for her scans, could also be used for this).

`run.py` operates on these black-and-white images of the individual characters, and runs the rest of the dataset preparation pipeline as explained in Figure 6 of my report. That means it:
- extracts the nominal 'CAD' shape from the font images in <FOLDER>, which Sara took from the dataset used in

Note: The code is very similar to the code I used to process the surgical guides. The files of the form _<>.py should be identical to those in the surgical guide GitHub repo.

## Getting Started

### Dependencies

I ran the code in Windows 11 Anaconda, using the modules contained in `environment.yml`. Also needed is pytorch (optionally alongside Nvidia CUDA).

Set up the environment by, in anaconda terminal in repository folder, executing `conda env create -f environment.yml` (this might take a while to install everything). It will create an environment called my_ccml_env, or whatever you change the first line to in `environment.yml`.
