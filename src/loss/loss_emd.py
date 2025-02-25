import numpy as np
import ot
try:
    from utils import * 
except:
    from  .utils import * 
import torch 
def compute_EMD(net,input_pc ,batch_size,dim_space=2,device='cuda'):
    # we compute L(F (Ni, fθ ), Nj ). 
    # We use the EMD loss to compute the distance between the two point clouds

    # sample batch size points and the corresponding normals
    idxs = alternative_random_choice(array=input_pc, size=batch_size, shape=0)
    sample_pc = input_pc[idxs]
    Ni=project_pc(net,sample_pc)
    Nj=input_pc[alternative_random_choice(array=input_pc, size=batch_size, shape=0)]

    # compute the EMD loss
    # Uniform weights for each point (assuming equal weight)
    weights1 = torch.ones(len(Ni)) / len(Ni)
    weights1 = weights1.to(device)
    weights2 = torch.ones(len(Nj))/ len(Nj)
    weights2= weights2.to(device)
    cost_matrix=ot.dist(Ni, Nj).to(device)
    # Compute the Earth Mover's Distance (Optimal Transport)
    emd_value = ot.emd2(weights1, weights2, cost_matrix)
    emd_value = emd_value.to(device)
    return emd_value

import geomloss
import torch
from geomloss import SamplesLoss
class EMD_loss(torch.nn.Module):
    def __init__(self):
        super(EMD_loss, self).__init__()
        self.loss=SamplesLoss(loss="sinkhorn")

    def forward(self, x, y):
        return self.loss(x, y)
def compute_EMD_geomloss(net,input_pc ,batch_size,loss,dim_space=2):
    # we compute L(F (Ni, fθ ), Nj ). 
    # We use the EMD loss to compute the distance between the two point clouds

    # sample batch size points and the corresponding normals
    idxs = alternative_random_choice(array=input_pc, size=batch_size, shape=0)
    sample_pc = input_pc[idxs]
    Ni=project_pc(net,sample_pc)
    Nj=input_pc[alternative_random_choice(array=input_pc, size=batch_size, shape=0)]

    # compute the EMD loss
    # Uniform weights for each point (assuming equal weight)   
    emd_value=loss(Ni, Nj)

    return emd_value