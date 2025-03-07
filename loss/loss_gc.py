import torch
import torch.nn as nn
try:
    from utils import * 
except:
    from  .utils import * 

#! not sure this is the right computation, ! On devrait sample aléatoirement dans l'espace! 
def loss_geometric_consistency(net,input_pc,batch_size=2000,dim_space=2):
    idxs = alternative_random_choice(array=input_pc, size=batch_size, shape=0)
    sample_pc = input_pc[idxs]
    # We compute λ|Ni| * R(E)
    ni_barre=input_pc.shape[0]
    d= net(input_pc)
    nprime=project_pc(net,input_pc=input_pc)
    # compute |fθ (n, c)| − minn′ ∈F (Ni,fθ ) ||n − n′||2)
    # Compute pairwise Euclidean distances in a vectorized way
    dist_matrix = torch.cdist(input_pc, nprime)  # Shape: (N, N)
    
    # Get the minimum distance for each point
    min_distances, _ = torch.min(dist_matrix, dim=1)
    # Apply R(E) and compute sum
    #Compute E
    E = torch.abs(torch.norm(d) - min_distances.unsqueeze(1))  # Broadcasting

    regularization_term = torch.sum(nn.ReLU()(E))

    return 1/input_pc.shape[0] *regularization_term if input_pc.shape[0] > 0 else torch.tensor(0.0)
def loss_geometric_consistency3(net,input_pc,batch_size=2000,dim_space=2):
    pts_random = sample_xt(batch_size, dim_space)
    pts_random.requires_grad = True

    
    nprime=project_pc(net,input_pc=pts_random)

    regularization_term = torch.linalg.norm(nprime,dim=1).mean()
    return regularization_term
# we compute the second term of the loss
#! not sure this is the right computation, ! On devrait sample aléatoirement dans l'espace! 
def loss_geometric_consistency2(net,input_pc,batch_size=2000,dim_space=2):
    pts_random = sample_xt(batch_size, dim_space)
    pts_random.requires_grad = True

    
    nprime=project_pc(net,input_pc=pts_random)
    # compute |fθ (n, c)| − minn′ ∈F (Ni,fθ ) ||n − n′||2)
    # Compute pairwise Euclidean distances in a vectorized way
    dist_matrix = torch.cdist(nprime, input_pc)  # Shape: (N, N)
    
    # Get the minimum distance for each point
    min_distances, _ = torch.min(dist_matrix, dim=1)
    # Apply R(E) and compute sum
    #Compute E
    # E = torch.abs(torch.norm(d) - min_distances.unsqueeze(1))  # Broadcasting
    E= min_distances.mean()


    return E if input_pc.shape[0] > 0 else torch.tensor(0.0)
