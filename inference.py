import os
import math
import numpy as np
import torch
import torch.nn as nn
import random
import copy
import nibabel as nib
import scipy.ndimage
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import Medical3D
from hypercube_sampling import hypercube_sampling
from MLP import MLP
from hypercube_rendering import hypercube_rendering
from loss import Adaptive_MSE_LOSS
from hypercube_imp import hypercube_imp
from RCAN4D import make_rcan4d
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage
from utils import *
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib  
import sys
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter


def inference(config_pat):


    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["is_train"] = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    latent_code_save_path = os.path.join(config["save_path"], "latent_code.pt")
    latent_code = torch.load(latent_code_save_path, map_location='cpu')
    latent_code = latent_code.to(device)
   
    evalloader, dataset = load_data(config, mode="eval")

    model_params = config["model"].copy()
    model_params["extra_ch"] = config["len_lz"]
    model1 = MLP(**model_params).to(device)

    model_checkpoint_path = os.path.join(config["save_path"], "current.pth")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    
    model_state = checkpoint["model1_state_dict"]
    model1.load_state_dict(model_state)

    model1.eval()

    predicted_slices = []
    
    new_X = int(dataset.x * dataset.scale)
    new_Y = int(dataset.y * dataset.scale)
    new_Z = int(dataset.z * dataset.scale)
    new_T = int(dataset.t * dataset.scaleT)

    prev_psnr_value = 1

    for idx, batch in tqdm(enumerate(evalloader),total=len(evalloader),desc="[Inferencing]"):

        coords, real_coords, data_shape, scale = batch
        coords = coords.squeeze(0).to(device)
        real_coords = real_coords.squeeze(0).to(device)

        concatenated_coords, loss_rcan, psnr_value = rcan_skip(
            latent_code, prev_psnr_value, real_coords, coords, device
        )

        pred_slice = predict_coords_slice(concatenated_coords, model1, device, config, new_Y, new_Z, new_T, hypercube_sampling, hypercube_rendering)
        predicted_slices.append(pred_slice)

    full_volume = torch.cat(predicted_slices, dim=0)
    print("Full volume shape before transpose:", full_volume.shape)

    full_volume_np = full_volume.numpy()

    v_min, v_max = np.percentile(full_volume_np, (0.1, 99.9))
    img_rescaled = rescale_intensity(full_volume_np, in_range=(v_min, v_max), out_range=(0, 255))
    img_rescaled = img_rescaled.astype(np.uint8)
    img_rescaled = gaussian_filter(img_rescaled, sigma=1)

    affine = np.eye(4) 
    nifti_img = nib.Nifti1Image(img_rescaled, affine)
    
    output_path = os.path.join(config["save_path"], "output.nii.gz")
    nib.save(nifti_img, output_path)
    print(f"Saved 4D NIfTI image to {output_path}")


if __name__ == "__main__":

    inference(config_path="./config/config.yaml")