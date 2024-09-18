import torchvision.models as models

model = models.googlenet(pretrained=True)
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print("Total memory for parameters: ", param_size)