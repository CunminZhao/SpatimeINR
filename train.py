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
import scipy.ndimage
from tqdm import tqdm
from skimage.exposure import rescale_intensity
from scipy.ndimage import gaussian_filter
import yaml

from Dataset.datasets import Medical3D
from Implicit_Rendering.hypercube_sampling import hypercube_sampling
from Model.MLP import MLP
from Implicit_Rendering.hypercube_rendering import hypercube_rendering
from Loss.loss import Adaptive_MSE_LOSS
from Model.RCAN4D import make_rcan4d
from utils import *

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["is_train"] = True

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    plot_enabled = False

    trainloader, dataset = load_data(config)
    data_iter = iter(trainloader)

    data_10, data_5 = downsample_img(config["dataset"]["train"]["file"], new_last_dim_data5=int(dataset.t / 2))
    data_tensor = torch.from_numpy(data_5).to(device)
    target_tensor = torch.from_numpy(data_10).unsqueeze(0).to(device)
    criterion = nn.MSELoss()

    model_params = config["model"].copy()
    model_params["extra_ch"] = config["len_lz"]
    model1 = MLP(**model_params).to(device)
    model1.train()

    rcan_params = config["rcan"].copy()
    rcan_params["len_Z"] = config["len_lz"]
    if rcan_params.get("in_channels") is None or rcan_params.get("out_channels") is None:
        rcan_params["in_channels"] = data_5.shape[-1]
        rcan_params["out_channels"] = data_10.shape[-1]
    rcan = make_rcan4d(**rcan_params).to(device)
    rcan.train()

    optimizer = torch.optim.Adam(
        list(model1.parameters()) + list(rcan.parameters()),
        lr=config["optim"]["lr"],
        betas=config["optim"]["betas"],
        eps=config["optim"]["eps"]
    )

    lr_init       = config["lr_decay"]["lr_init"]
    lr_final      = config["lr_decay"]["lr_final"]
    max_iter      = config["lr_decay"]["max_iter"]
    lr_delay_steps = config["lr_decay"]["lr_delay_steps"]
    lr_delay_mult  = config["lr_decay"]["lr_delay_mult"]
    clip_max_val  = config["clip_grad"]["max_val"]
    clip_max_norm = config["clip_grad"]["max_norm"]

    if plot_enabled:
        plt.ion()
        fig, ax1, ax2 = init_plots()
    else:
        fig, ax1, ax2 = None, None, None

    avg_loss_history = []
    avg_psnr_history = []
    iter_history = []
    
    interval_losses = []
    interval_psnrs = []

    prev_psnr_value = 1

    pbar = tqdm(range(config["max_iter"]), desc="Training")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainloader)
            batch = next(data_iter)

        data, coords, real_coords, data_shape, scale = batch
        data = data.squeeze(0).to(device)
        coords = coords.squeeze(0).to(device)
        real_coords = real_coords.squeeze(0).to(device)

        if math.floor(prev_psnr_value) < config["threshold"]:
            loss_rcan, psnr_value, latent_code = rcan_forward(
                rcan, data_tensor, target_tensor, criterion
            )
            prev_psnr_value = psnr_value
            loss_MLP = torch.tensor(0.0, device=device)
            psnr_display = 1.0
        else:
            concatenated_coords, loss_rcan, psnr_value = rcan_skip(
                latent_code, prev_psnr_value, real_coords, coords, device
            )
            samp_params = {
                "batch": concatenated_coords,
                "n_samples": config["n_samples"],
                "len_lz": config["len_lz"],
                "is_train": config["is_train"]
            }
            sample_ans = hypercube_sampling(**samp_params)
            pts = sample_ans["pts"].to(device)
            raw0 = model1(pts)
            out0 = hypercube_rendering(raw0, **sample_ans)
            loss_MLP = criterion(out0["rgb"].to(device), data)
            psnr_display = 10 * math.log10(1 / (loss_MLP.item() + 1e-8))

        total_loss = loss_MLP + loss_rcan
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad(optimizer, clip_max_val, clip_max_norm)
        optimizer.step()

        torch.cuda.empty_cache()

        current_lr = lr_decay(optimizer, step, lr_init, lr_final, max_iter, lr_delay_steps, lr_delay_mult)
        
        interval_losses.append(total_loss.item())
        interval_psnrs.append(psnr_display)

        if (step + 1) % config["log_iter"] == 0:
            avg_loss = sum(interval_losses) / len(interval_losses)
            avg_psnr = sum(interval_psnrs) / len(interval_psnrs)

            pbar.set_postfix({
                "Avg Loss": f"{avg_loss:.6f}",
                "Avg PSNR": f"{avg_psnr:.6f}",
                "lr": f"{current_lr:.6f}"
            })

            current_iter = step + 1
            iter_history.append(current_iter)
            avg_loss_history.append(avg_loss)
            avg_psnr_history.append(avg_psnr)
            
            if plot_enabled:
                update_plots(ax1, ax2, iter_history, avg_loss_history, avg_psnr_history)
            interval_losses = []
            interval_psnrs = []

        if (step + 1) % config["save_iter"] == 0:
            save_path = os.path.join(config["save_path"], "current.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_dict = {
                "model1_state_dict": model1.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step + 1
            }
            torch.save(save_dict, save_path)

            latent_code_to_save = latent_code.detach().cpu()
            latent_code_save_path = os.path.join(config["save_path"], "latent_code.pt")
            torch.save(latent_code_to_save, latent_code_save_path)

            print("Model saved as current.pth")

    if plot_enabled:
        plt.ioff()
        plt.show()

    print("Model saved as current.pth")

if __name__ == "__main__":
    config = parse_arguments_and_update_config()
    train(config)