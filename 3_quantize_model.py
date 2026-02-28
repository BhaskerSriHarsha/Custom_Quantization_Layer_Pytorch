import torch
from base_model_def import base_model, newLayer
import os

DEVICE = "cuda:0"

base_m = base_model().to(DEVICE)
base_m.load_state_dict(torch.load("base_saved_model.pth", map_location=DEVICE, weights_only=True))

for name, i in base_m.named_parameters():
    W = i.data
    max_w = torch.max(abs(W)).item()
    s = max_w/127.0

    W_q = torch.clamp(torch.round(W / s), min=-127, max=127).to(torch.int8)

    newl = newLayer()
    newl.q_layer.copy_(W_q)
    newl.weight_scale.fill_(s)

    base_m.layer = newl
    print("Swapped the old layer with the new layer")

    for name, i in base_m.named_buffers():
        print(name, i)

    torch.save(base_m.state_dict(), "quantized_model.pth")
    print("Saved the quantized model to disk!")


print(f"\n\nVerification of sizes: ")
print(f"Original: {os.path.getsize("base_saved_model.pth")/1024:.1f} KB")
print(f"Quantized: {os.path.getsize("quantized_model.pth")/1024:.1f} KB")