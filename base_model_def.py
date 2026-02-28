import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.register_buffer('weight_scale', torch.ones(1, dtype=torch.float32))

    def forward(self, x):
        # On-the-fly dequantization: W_approx = W_q * s
        W_approx = self.q_layer.to(torch.float32) * self.weight_scale
        
        # Standard FP32 matrix multiplication
        return F.linear(x, W_approx)
    
class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Notice we instantiate newLayer() here, not nn.Linear()
        self.layer = newLayer() 
        
    def forward(self, x):
        x = x.view(-1, 784)
        return self.layer(x)