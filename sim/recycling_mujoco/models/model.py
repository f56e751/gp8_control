import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, dimensions):
        super(FCN, self).__init__()
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i == len(dimensions) - 2:
                break
            layers.append(nn.ReLU())
            
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)