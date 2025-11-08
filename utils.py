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
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.ndimage
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter


from Dataset.datasets import Medical3D
from Implicit_Rendering.hypercube_sampling import hypercube_sampling
from Model.MLP import MLP
from Implicit_Rendering.hypercube_rendering import hypercube_rendering
from Loss.loss import Adaptive_MSE_LOSS
from Model.RCAN4D import make_rcan4d
from utils import *

import yaml



def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
    

def downsample_img(file_path, new_last_dim_data5=5):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        print("raw shape:", data.shape)
        
        original_t = data.shape[-1]
        spatial_zoom = (0.5, 0.5, 0.5)
        zoom_factors_data10 = spatial_zoom + (1.0,)
        data_10 = scipy.ndimage.zoom(data, zoom_factors_data10, order=3)
        
        t_zoom_factor = new_last_dim_data5 / original_t
        zoom_factors_data5 = spatial_zoom + (t_zoom_factor,)
        data_5 = scipy.ndimage.zoom(data, zoom_factors_data5, order=3)

        data_10=(data_10 - data_10.min()) / (data_10.max() - data_10.min())
        data_5=(data_5 - data_5.min()) / (data_5.max() - data_5.min())
        
        
        return data_10.astype(np.float32), data_5.astype(np.float32)
        
    except Exception as e:
        print(e)



def rcan_forward(rcan, data_tensor, target_tensor, criterion):

    inp = data_tensor.unsqueeze(0)
    out, img_representation = rcan(inp)
    loss_rcan = criterion(out, target_tensor)
    psnr_value = calculate_psnr(out, target_tensor)

    if img_representation.dim() == 5:
        img_representation = img_representation.squeeze(0)


    return loss_rcan, psnr_value, img_representation.detach()


def rcan_skip(prev_img_representation, prev_psnr_value, real_coords, coords, device):

    loss_rcan = torch.tensor(0.0, device=device)
    psnr_value = prev_psnr_value
    img_representation = prev_img_representation

    if img_representation.dim() == 5:
        img_representation = img_representation.squeeze(0)

    x_idx = real_coords[:, 0]
    y_idx = real_coords[:, 1]
    z_idx = real_coords[:, 2]
    lz = img_representation[x_idx, y_idx, z_idx, :].to(device)

    coords = coords.to(device)
    concatenated_coords = torch.cat((coords, lz), dim=1)

    return concatenated_coords, loss_rcan, psnr_value


def calculate_psnr(pred, target, max_val=None):

    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    if max_val is None:
        max_val = torch.max(target)
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()


def clip_grad(optimizer, max_val, max_norm):

    for param_group in optimizer.param_groups:
        if max_val > 0:
            torch.nn.utils.clip_grad_value_(param_group['params'], max_val)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm)

def lr_decay(optimizer, step, lr_init, lr_final, max_iter, lr_delay_steps=0, lr_delay_mult=1):

    def log_lerp(t, v0, v1):
        lv0 = np.log(v0)
        lv1 = np.log(v1)
        return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)

    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
    else:
        delay_rate = 1.0

    new_lr = delay_rate * log_lerp(step / max_iter, lr_init, lr_final)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def load_data(config, mode="train"):

    dataset_params = config["dataset"][mode].copy()
    dataset_params["mode"] = mode
    dataset = Medical3D(**dataset_params)

    dataloader_params = config["dataloader"][mode]
    trainloader = DataLoader(
        dataset, 
        batch_size=dataloader_params["batch_size"], 
        shuffle=dataloader_params["shuffle"]
    )
    return trainloader, dataset


def predict_coords_slice(coords, mlp, device, config, new_Y, new_Z, new_T, hypercube_sampling, hypercube_rendering):

    with torch.no_grad():
        chunk_size = config["dataset"]["eval"]["bsize"] * 5
        pred_chunks = []
        coords_device = coords.to(device)
        
        for i in range(0, coords_device.shape[0], chunk_size):
            chunk = coords_device[i:i+chunk_size]

            samp_params = {
                "batch": chunk,
                "n_samples": config["n_samples"],
                "len_lz": config["len_lz"],
                "is_train": config["is_train"]
            }

            sample_ans = hypercube_sampling(**samp_params)
            pts = sample_ans["pts"].to(device)
            raw0 = mlp(pts)
            out0 = hypercube_rendering(raw0, **sample_ans)
            pred_chunk = out0["rgb"]

            pred_chunks.append(pred_chunk.cpu())

        pred = torch.cat(pred_chunks, dim=0)
    pred = pred.squeeze(-1)
    pred_slice = pred.view(new_Y, new_Z, new_T).unsqueeze(0)
    return pred_slice


def init_plots():

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    ax1.set_title('Loss over iterations')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax2.set_title('PSNR over iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR')
    plt.tight_layout()
    return fig, ax1, ax2


def update_plots(ax1, ax2, iter_history, avg_loss_history, avg_psnr_history):

    ax1.cla()
    ax1.plot(iter_history, avg_loss_history, 'r-', label='Avg Loss')
    ax1.set_title('Loss over iterations')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.cla()
    ax2.plot(iter_history, avg_psnr_history, 'g-', label='Avg PSNR')
    ax2.set_title('PSNR over iterations')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR')
    ax2.legend()

    plt.tight_layout()
    plt.pause(0.001)


def parse_arguments_and_update_config():

    import argparse
    parser = argparse.ArgumentParser(description="Update configuration and run inference")
    parser.add_argument("--config", type=str, default="./Config/config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--save_path", type=str,
                        help="Override the 'save_path' in the configuration; e.g., './save39'")
    parser.add_argument("--file", type=str,
                        help="Override 'dataset.train.file' and 'dataset.eval.file' in the configuration; e.g., '../data/01_4d_10.nii.gz'")
    parser.add_argument("--scale", type=int,
                        help="Override the 'dataset.eval.scale' in the configuration; e.g., 1")
    parser.add_argument("--scaleT", type=int,
                        help="Override the 'dataset.eval.scaleT' in the configuration; e.g., 2")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.save_path is not None:
        config["save_path"] = args.save_path
    if args.file is not None:
        if "dataset" in config:
            if "train" in config["dataset"]:
                config["dataset"]["train"]["file"] = args.file
            if "eval" in config["dataset"]:
                config["dataset"]["eval"]["file"] = args.file
    if args.scale is not None:
        if "dataset" in config and "eval" in config["dataset"]:
            config["dataset"]["eval"]["scale"] = args.scale
    if args.scaleT is not None:
        if "dataset" in config and "eval" in config["dataset"]:
            config["dataset"]["eval"]["scaleT"] = args.scaleT

    return config