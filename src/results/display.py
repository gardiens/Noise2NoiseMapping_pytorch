import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def display_result(f, resolution, filename=None, figsize=(14, 5),title="",eps=0.005):
    """
    Displays the values of the function f, evaluated over a regular grid 
    defined between -1 and 1 with a resolution of (resolution x resolution).
    """
    fig, ax = plt.subplots(figsize=figsize)  # Fix subplot creation
    time = 1
    i = time // 5
    j = time - i * 5
    t = 0.1 * time

    # Define device properly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a grid of coordinates
    coords = torch.stack(list(torch.meshgrid([torch.linspace(-1, 1, resolution, device=device)] * 2, indexing='xy')), dim=2)
    coords = coords.reshape(-1, 2)  # Shape: (resolution*resolution, 2)
    print("coords shape before adding t:", coords.shape)

    # Ensure coordinates require gradients
    coords.requires_grad = True

    # Forward pass through the network
    sdf = f(coords)

    # Fix reshaping: Remove singleton dimension if present
    sdf = sdf.squeeze(-1).reshape(resolution, resolution)
    numpy_sdf = sdf.detach().cpu().numpy()

    # Processing SDF for visualization
    numpy_sdf_max = 1 - np.maximum(numpy_sdf, np.zeros(numpy_sdf.shape))
    numpy_sdf_max = numpy_sdf_max - np.multiply(numpy_sdf_max, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps))

    numpy_sdf_min = 1 - np.maximum(-numpy_sdf, np.zeros(numpy_sdf.shape))
    numpy_sdf_min = numpy_sdf_min - np.multiply(numpy_sdf_min, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps))

    numpy_sdf_both = 1 - np.maximum(numpy_sdf, np.zeros(numpy_sdf.shape)) - np.maximum(-numpy_sdf, np.zeros(numpy_sdf.shape))
    numpy_sdf_both = numpy_sdf_both - np.multiply(numpy_sdf_both, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps))

    # Displaying the result
    ax.axis('off')
    ax.imshow(np.concatenate([numpy_sdf_min[:, :, np.newaxis], 
                              numpy_sdf_both[:, :, np.newaxis], 
                              numpy_sdf_max[:, :, np.newaxis]], axis=2))
    ax.contour(numpy_sdf, 10, colors='k', linewidths=0.4, linestyles='solid')

    # Show or save the figure
    plt.title(title)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
