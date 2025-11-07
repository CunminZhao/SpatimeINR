import torch
import math

def hypercube_sampling(batch, len_lz, n_samples, is_train):
    data_main, extra_feats = torch.split(batch, [12, len_lz], dim=-1)
    cnts, x_bound, y_bound, z_bound, t_bound = torch.split(data_main, [4, 2, 2, 2, 2], dim=-1)
    base_steps = round(math.pow(n_samples, 1.0 / 4))
    steps = base_steps + 2  
    total_points_generated = steps ** 4    
    grid = torch.linspace(0., 1., steps=steps, device=batch.device)
    mesh_x, mesh_y, mesh_z, mesh_t = torch.meshgrid(grid, grid, grid, grid, indexing='ij')
    t_vals_full = torch.stack([mesh_x, mesh_y, mesh_z, mesh_t], dim=-1)
    t_vals_interior = t_vals_full[1:-1, 1:-1, 1:-1, 1:-1]
    t_vals = t_vals_interior.contiguous().view(-1, 4)
    t_vals = t_vals.unsqueeze(0)
    
    B = batch.shape[0]
    n_points = t_vals.shape[1]    
    x_l = x_bound[:, 0:1].unsqueeze(1) 
    x_r = x_bound[:, 1:2].unsqueeze(1)    
    y_l = y_bound[:, 0:1].unsqueeze(1)
    y_r = y_bound[:, 1:2].unsqueeze(1)    
    z_l = z_bound[:, 0:1].unsqueeze(1)
    z_r = z_bound[:, 1:2].unsqueeze(1)    
    t_l = t_bound[:, 0:1].unsqueeze(1)
    t_r = t_bound[:, 1:2].unsqueeze(1)

    if is_train:
        x_rand = torch.rand(B, n_points, 1, device=batch.device)
        y_rand = torch.rand(B, n_points, 1, device=batch.device)
        z_rand = torch.rand(B, n_points, 1, device=batch.device)
        t_rand = torch.rand(B, n_points, 1, device=batch.device)
        
        x_vals = x_l + t_vals[:, :, 0:1] * (x_r - x_l) * x_rand
        y_vals = y_l + t_vals[:, :, 1:2] * (y_r - y_l) * y_rand
        z_vals = z_l + t_vals[:, :, 2:3] * (z_r - z_l) * z_rand
        t_vals_coord = t_l + t_vals[:, :, 3:4] * (t_r - t_l) * t_rand
    else:
        x_vals = x_l + t_vals[:, :, 0:1] * (x_r - x_l)
        y_vals = y_l + t_vals[:, :, 1:2] * (y_r - y_l)
        z_vals = z_l + t_vals[:, :, 2:3] * (z_r - z_l)
        t_vals_coord = t_l + t_vals[:, :, 3:4] * (t_r - t_l)
    
    pts_coords = torch.cat([x_vals, y_vals, z_vals, t_vals_coord], dim=-1)    
    extra_feats_expanded = extra_feats.unsqueeze(1).expand(B, n_points, len_lz)
    pts = torch.cat([pts_coords, extra_feats_expanded], dim=-1)
    
    dx = (x_r - x_l).mean() / 2
    dy = (y_r - y_l).mean() / 2
    dz = (z_r - z_l).mean() / 2
    dt = (t_r - t_l).mean() / 2
    
    return {
        'pts': pts,
        'cnts': cnts,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'dt': dt
    }

if __name__ == '__main__':