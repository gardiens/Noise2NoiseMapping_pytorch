import torch
import torch.nn as nn
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
from torch import autograd

def nabla(model,x):
    with torch.enable_grad():
        x= x.requires_grad_(True)
        y=model(x)
        nablas=autograd.grad(
            y,
            x,
            torch.ones_like(y,device=x.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    return nablas
def project_pc(net,input_pc):
    # apply F (n, fθ ) = n − d × ∇fθ (n, c)/||∇fθ (n, c)||2
    d= net(input_pc)
    # grad= gradient(d, input_pc)
    grad=nabla(model=net,x=input_pc)
    return input_pc - d * grad/ torch.norm(grad, dim=1).view(-1,1)


def alternative_random_choice(
    array: "torch.tensor", size: int, replace: bool = False, shape: int = 0
):
    size= min(size, array.shape[shape])
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