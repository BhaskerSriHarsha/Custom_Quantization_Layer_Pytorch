import torch
import torch.nn as nn

class base_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784, 10, bias=False)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.layer(x)
    
class newLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("q_layer", torch.ones((10,784), dtype=torch.int8))
    def forward(self, x):
        pass # Write the dequantize + forward pass here