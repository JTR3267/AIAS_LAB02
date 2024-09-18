import torchvision.models as models

model = models.googlenet(pretrained=True)
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)