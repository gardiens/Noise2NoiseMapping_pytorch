import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
def display_projection(net, input_pc):
    """
    Projects the input point cloud onto a learned surface using the SDF network 
    and displays the projected points in 2D.
    """
    # Compute signed distance and gradient
    from src.loss.utils import gradient
    d = net(input_pc)
    grad = gradient(d, input_pc)
    
    # Compute the projected points
    S_projected = input_pc - d * grad / torch.norm(grad, dim=1, keepdim=True)

    # Display as 2D scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))  # Fix subplot initialization
    pc_numpy = S_projected.detach().cpu().numpy()

    ax.scatter(pc_numpy[:, 0], pc_numpy[:, 1])
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_title("Projected Points")

    plt.show()
def display_result_3D(f, resolution=100, filename=None, figsize=(14, 5), title="3D SDF Result", eps=0.0):
    device = next(f.parameters()).device  # get device from network
    # Create a 3D grid of points in the cube [-1, 1]^3
    lin = torch.linspace(-1, 1, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    coords.requires_grad = True

    # Forward pass through the network
    sdf = f(coords).squeeze()
    sdf_volume = sdf.reshape(resolution, resolution, resolution).detach().cpu().numpy()

    # Option: Display slices along a chosen axis (e.g., z-axis)
    import matplotlib.pyplot as plt
    slice_axis = 2
    num_slices = 5
    slice_indices = np.linspace(0, resolution-1, num_slices, dtype=int)
    fig, axes = plt.subplots(1, num_slices, figsize=figsize)
    for i, idx in enumerate(slice_indices):
        slice_img = sdf_volume[:, :, idx]  # slice along z-axis
        im = axes[i].imshow(slice_img, cmap='viridis', origin='lower', extent=[-1,1,-1,1])
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis("off")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


def display_loss(list_loss,skip_keys=[]):
    plt.figure(figsize=(6, 4))
    plt.yscale("log")
    for keys in list_loss.keys():
        
        loss=list_loss[keys]
        if len(loss)==0 or keys in skip_keys:
            continue
        print("keys",keys,loss)
        plt.plot(loss,label="loss:{} ({:.2f})".format(keys,loss[-1]))
    plt.xlabel("Epochs")
    plt.legend()