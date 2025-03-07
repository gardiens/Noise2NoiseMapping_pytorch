import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def sphere_data(npts, nb_hints, resolution=50,dim=2):
    # Generate npts random points in 3D and project them onto the surface of a sphere of radius 0.5
    pts = torch.randn(npts, dim, device=device)
    pc = 0.5 * torch.nn.functional.normalize(pts, dim=1)
    # For normals, you could use the original random directions (or recompute from pc)
    nc = torch.nn.functional.normalize(pts, dim=1)
    
    # Generate hint points uniformly in the cube [-1, 1]^3.
    pts_hint = torch.rand(nb_hints, 3, device=device) * 2 - 1
    # Ground truth SDF for a sphere: distance from the point to the sphere of radius 0.5
    gt_sdf_hint = torch.norm(pts_hint, dim=1) - 0.5
    # For gradients, the true gradient at a point (away from the sphere) is the normalized vector.
    gt_grad_hint = torch.nn.functional.normalize(pts_hint, dim=1)
    
    # Create a 3D grid for the SDF volume.
    lin = torch.linspace(-1, 1, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # shape: [resolution, resolution, resolution, 3]
    # Compute the ground truth SDF at each grid point.
    gt_coords = torch.norm(coords, dim=-1) - 0.5  # shape: [resolution, resolution, resolution]
    
    return pc, nc, pts_hint, gt_sdf_hint, gt_grad_hint, coords, gt_coords



def sdf_square(pts):
    d_ext = torch.norm(torch.maximum(torch.abs(pts)-0.5,torch.zeros_like(pts)),2,dim=-1)
    d_int = -torch.min(torch.abs(torch.minimum(torch.abs(pts)-0.5,torch.zeros_like(pts))),dim=-1)[0]
    gt_sdf = d_ext+d_int
    return gt_sdf, d_ext, d_int

def square_data(npts, nb_hints, resolution=200) :
    #points on the surface (square of radius 0.5, time between 0 and 1)
    np = npts // 4

    pc1 = torch.cat((torch.ones(np,1,device=device)*0.5,torch.rand(np,1,device=device)-0.5),1)
    nc1 = torch.cat((torch.ones(np,1,device=device),torch.zeros(np,1,device=device)),1)

    pc2 = torch.cat((torch.ones(np,1,device=device)*(-0.5),torch.rand(np,1,device=device)-0.5),1)
    nc2 = torch.cat((-torch.ones(np,1,device=device),-torch.zeros(np,1,device=device)),1)

    pc3 = torch.cat((torch.rand(np,1,device=device)-0.5, torch.ones(np,1,device=device)*0.5),1)
    nc3 = torch.cat((torch.zeros(np,1,device=device), torch.ones(np,1,device=device)),1)

    pc4 = torch.cat((torch.rand(np,1,device=device)-0.5, torch.ones(np,1,device=device)*(-0.5)),1)
    nc4 = torch.cat((torch.zeros(np,1,device=device), -torch.ones(np,1,device=device)),1)

    pc = torch.cat((pc1, pc2, pc3, pc4),0)
    nc = torch.cat((nc1, nc2, nc3, nc4),0)

    hints = torch.rand(nb_hints,2, device=device)*2 - 1
    
    grad_ext = torch.maximum(torch.abs(hints)-0.5,torch.zeros_like(hints))
    grad_ext = torch.nn.functional.normalize(grad_ext)*((hints > 0)*2-1)
    
    ind = torch.min(torch.abs(torch.minimum(torch.abs(hints)-0.5,torch.zeros_like(hints))),dim=1)[1].unsqueeze(1)

    gt_sdf_hint, d_ext, d_int = sdf_square(hints)
    indices = (1 - (d_ext > 0)*1).unsqueeze(1)
    indices = torch.cat((indices, indices), dim=1)
    grad_int = torch.cat((1-ind,ind),dim=1)*((hints > 0)*2-1)*indices

    
    gt_grad_hint = grad_ext + grad_int
    pts_hint = hints

    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device = device)]*2, indexing = 'xy')), dim=2)#.reshape(-1, 2)
    gt_coords,_, _ = sdf_square(coords)

    return pc,nc,pts_hint,gt_sdf_hint,gt_grad_hint, coords, gt_coords

def cst_radialVF(pc):
    vf = torch.nn.functional.normalize(pc[:,0:2])
    return vf
