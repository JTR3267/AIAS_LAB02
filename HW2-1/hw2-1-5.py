import torch
import torch.nn as nn
import torchvision.models as models

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = models.googlenet(pretrained=True)
activation = {}

for layer_name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        # 註冊 Conv2d layer
        layer.register_forward_hook(get_activation(layer_name))

data = torch.randn(1, 3, 224, 224)
output = model(data)

for layer in activation:
    print(f"Activation from layer {layer}: {activation[layer]}")