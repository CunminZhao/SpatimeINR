import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class baseModel(nn.Module):
    def __init__(self, params):
        super(baseModel, self).__init__()
        for k, v in params.items():
            setattr(self, k, v)
        self.freqs = 2. ** torch.linspace(0., self.max_freq, steps=self.max_freq + 1)
        self.embed_dim = self.in_ch * (len(self.p_fns) * (self.max_freq + 1) + 1)

    def embed(self, coords):

        embedded = torch.cat(
            [coords] + [getattr(torch, p_fn)(coords * freq)
                        for freq in self.freqs
                        for p_fn in self.p_fns],
            dim=-1
        )
        return embedded

class MLP(baseModel):
    def __init__(self, **params):

        self.extra_ch = params.pop('extra_ch', 0)
        super(MLP, self).__init__(params)
        
        total_in_dim = self.embed_dim 
        
        self.netD = params['netD']
        self.netW = params['netW']
        self.out_ch = params['out_ch']
        
        self.coords_MLP = nn.ModuleList()
        self.coords_MLP.append(nn.Linear(total_in_dim, self.netW))
        
        for i in range(self.netD - 1):
            if (i + 1) in self.skips:

                layer = nn.Linear(self.netW + total_in_dim, self.netW)
            else:
                layer = nn.Linear(self.netW, self.netW)
            self.coords_MLP.append(layer)
        
        self.out_MLP = nn.Linear(self.netW, self.out_ch)



    def forward(self, x):

        coords = x[..., :self.in_ch]   
        extra = x[..., self.in_ch:]      

        emb_coords = self.embed(coords)
        

        full_input = emb_coords + extra

        h = full_input

        for idx, mlp in enumerate(self.coords_MLP):
            if idx in self.skips:
                #print("skips",idx, mlp)
                h = torch.cat([full_input, h], dim=-1)
                h = F.relu(mlp(h))
            else:
                #print(idx, mlp)
                h = F.relu(mlp(h))
    
        out = self.out_MLP(h)
        return out
