import torch
import torch.nn as nn

class SDFNet(nn.Module):
    def __init__(self, ninputchannels, dropout=0.2, gamma=0, sal_init=False, eik=False):
        super(SDFNet, self).__init__()
        ## Prepare the layers
        ## Don't forget to initialize your weights correctly.

        ## gamma, sal_init, eik are for later
        self.gamma=gamma
        self.eik = eik

        self.fc1 = nn.Linear(ninputchannels, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_weights()

        


        #custom weights init
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        return self
    def forward(self,x):
        ## Logic of the neural network
        ## You can add dropout if you want

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x=self.tanh(x)

        return x
