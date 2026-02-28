import torch
import torch.nn as nn

class base_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784, 10, bias=False)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.layer(x)