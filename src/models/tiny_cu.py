import torch 
import tinycudann as tcnn
import numpy as np
import torch.nn as nn

class SDF_TCNN(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 fix_init=False,
                 Xavier=True):
        super().__init__()
        print("initialisation Xavier",Xavier,"fix_init",fix_init)
        if  fix_init and Xavier:
            return ValueError( " Xavier and fixed initialization cannot be both True")
        self.xavier=Xavier
        self.fix_init=fix_init
        self.in_features=input_dim
        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        
        assert self.skips == [], 'TCNN does not support concatenating inside, please use skips=[].'

        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )


        self.backbone= nn.Sequential(
            nn.Linear(self.encoder.n_output_dims, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),

            nn.Linear(hidden_dim, 1)
        )

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            
            size = np.sqrt(6/self.in_features)/30
            size=1
            # for now size=1 seems to work better
            # print("size",size)
            ## Other initialization
            for layer in self.backbone:
                if isinstance(layer, nn.Linear):
                    #layer.weight.uniform_(-size, size)
                    # init with xavier glorito
                    if self.xavier:
                        nn.init.xavier_uniform_(layer.weight)
                    if self.fix_init:
                        layer.weight.uniform_(-size, size)
    def forward(self, x):
        # x: [B, 3]

        x = (x + 1) / 2 # to [0, 1]
        # print("initial x",x)
        x = self.encoder(x)
        # print("x dtye",x.dtype)
        # convert x to float 32
        # print("x.shape",x.shape)
        # check the type of backbone
        # print("backbone",self.backbone[0].weight.dtype)
        # print("output of x",torch.unique(x))
        
        x = x.to(dtype=torch.float)
        # print("after",x)
        h = self.backbone(x)

        if self.clip_sdf is not None:
            # print("did we clamp it?")
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h