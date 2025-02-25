import torch
import torch.nn as nn
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
def project_pc(net,input_pc):
    # apply F (n, fθ ) = n − d × ∇fθ (n, c)/||∇fθ (n, c)||2
    d= net(input_pc)
    grad= gradient(d, input_pc)
    return input_pc - d * grad/ torch.norm(grad, dim=1).view(-1,1)


def alternative_random_choice(
    array: "torch.tensor", size: int, replace: bool = False, shape: int = 0
):
    weights = torch.ones(array.shape[shape])

    idx = torch.multinomial(weights, size, replacement=replace)
    return idx 
import torch
import torch.nn as nn

def sample_xt(batch_size, dim_space=2,device='cuda'): #!
    ## Sample random points in space ([-1, 1]^2) and in time ([0, 1])
    list = [batch_size, dim_space]
    pts_random = (torch.rand(list, device=device)) * 2 - 1
    time_random = torch.rand(batch_size, device=device)

    return pts_random