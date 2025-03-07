import torch
import torch.nn as nn

try:
    from utils import alternative_random_choice
except:
    from  .utils import alternative_random_choice
# data should be 0 on the shape
def loss_data(sdf_pc):
    return torch.mean(sdf_pc**2)  # 

def loss_normalign(grad_pc, sample_nc):
    first_ps = torch.sum(sample_nc * grad_pc, dim=1)
    return ((1 - first_ps) ** 2).mean()

def loss_shape_data(net, pc, normals=None, batch_size=2000, dim_space=2):
    # sample batch size points and the corresponding normals
    idxs = alternative_random_choice(array=pc, size=batch_size, shape=0)
    sample_pc = pc[idxs]
    # sample_pc.requires_grad = True
    # sample_nc = normals[idxs]  #! Same idx !

    sdf_pc = net(sample_pc)
    # spatial gradients
    # grad_pc = gradient(sdf_pc, sample_pc)[:, 0:2]

    ## compute loss
    loss_pc = 100 * loss_data(sdf_pc) # + loss_normalign(grad_pc, sample_nc)
    return loss_pc

def loss_amb(net, pc_hint, gt_sdf, batch_size=2000):
    ## Sample random points in pc_hint and compute the SDF loss (as in previous lab but no clamp)
    idxs = alternative_random_choice(array=pc_hint, size=batch_size, shape=0)
    sample_pc = pc_hint[idxs]
    sample_pc.requires_grad = True
    sdf_pc = net(sample_pc)
    sample_gt_sdf = gt_sdf[idxs]
    loss_lamb = torch.mean((sdf_pc - sample_gt_sdf) ** 2)
    return 10 * loss_lamb
