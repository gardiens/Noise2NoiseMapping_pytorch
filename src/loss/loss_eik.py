try:
    from utils import * 
except:
    from  .utils import * 
def loss_eikonal_pts(grad_pts):
    ## Get the eikonal loss given spatial gradients
    return torch.mean(torch.abs(torch.norm(grad_pts, dim=1) - 1))

def loss_eikonal(net, batch_size, dim_space=2):
    ## Sample random points in space ([-1, 1]^2) and in time ([0, 1]), and compute eikonal loss
    pts_random = sample_xt(batch_size, dim_space)
    pts_random.requires_grad = True

    sdf_random = net(pts_random)

    grad_tot_random = gradient(sdf_random, pts_random)
    grad_spatial = grad_tot_random[:, 0:dim_space]
    return loss_eikonal_pts(grad_spatial)
