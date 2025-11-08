import torch
import math
import torch.nn.functional as F

def hypercube_rendering(raw, pts, cnts, dx, dy, dz, dt):

    pts22 = pts[:, :, :4]
    norm = torch.sqrt(torch.square(dx) + torch.square(dy) + torch.square(dz) + torch.square(dt))    
    raw2beta = lambda raw, dists, rs: -F.relu(raw) * dists * torch.square(rs) * 4 * math.pi
    raw2alpha = lambda raw, dists, rs: (1. - torch.exp(-F.relu(raw) * dists)) * torch.square(rs) * 4 * math.pi     
    rs = torch.norm(pts22 - cnts[:, None], dim=-1)     
    sorted_rs, indices_rs = torch.sort(rs, dim=-1)    
    dists = sorted_rs[..., 1:] - sorted_rs[..., :-1]
    dists = torch.cat([dists, dists[..., -1:]], dim=-1)     
    rgb = torch.gather(torch.sigmoid(raw[..., :-1]), dim=-2, index=indices_rs[..., None].expand_as(raw[..., :-1]))
    sorted_raw = torch.gather(raw[..., -1], dim=-1, index=indices_rs)
    beta = raw2beta(sorted_raw, dists, sorted_rs / norm)
    alpha = raw2alpha(sorted_raw, dists, sorted_rs / norm)  
    beta_cat = torch.cat([torch.zeros(alpha.shape[0], 1, device=alpha.device), beta], dim=-1)
    cumulative = torch.cumsum(beta_cat, dim=-1)[:, :-1]
    weights = alpha * torch.exp(cumulative)
    rgb_map = torch.sum(weights * rgb.squeeze(-1), dim=-1)
    
    return {'rgb': rgb_map, 'weights': weights, 'indices_rs': indices_rs}


if __name__ == '__main__':
    pass
    

