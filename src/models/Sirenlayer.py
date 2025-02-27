import torch 
import torch.nn as nn
import numpy as np

class SirenLayer(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        ## Create the layer, and initialize it. You can do it in init_weights

        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                ## Initialization of first layer type
                size = 1 / self.in_features
            else:
                size = np.sqrt(6 / self.in_features) / self.omega_0

                ## Other initialization
            self.fc1.weight.uniform_(-size, size)

    def forward(self, input):
        ## Logic
        return torch.sin(self.omega_0 * self.fc1(input))

class SirenNet(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, skip=[], omega_0=30.0):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.skip = [i in skip for i in range(num_layers)]
        self.omega_0 = omega_0
        layers_list = []
        ## Create layer
        # first layer
        layers_list.append(
            SirenLayer(
                in_features=dim_in,
                out_features=dim_hidden,
                is_first=True,
                omega_0=omega_0,
            )
        )
        for k in range(1, num_layers):
            layers_list.append(
                SirenLayer(
                    in_features=dim_hidden,
                    out_features=dim_hidden,
                    is_first=False,
                    omega_0=omega_0,
                )
            )
        ## Last layer is a simple linear layer. Don't forget to intialize your weights as before!
        self.skip.append(False)
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=True)
        ## Init last layer
        size = np.sqrt(6 / dim_in) / self.omega_0
        with torch.no_grad():
            self.last_layer.weight.uniform_(-size, size)
        # create the network
        self.layers = nn.ModuleList(layers_list)

    def forward(self, x):
        ## Network logic
        ## You can ignore skip connections at the beginning
        #! The skip is never initialized so we don't know what is exactly this parameters,
        #! we assume it's a list of index where we have to skip the connection

        for index, layer in enumerate(self.layers):
            if self.skip[index]:
                x = x + layer(x)
            else:
                x = layer(x)
        output = self.last_layer(x)
        return output
