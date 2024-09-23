import torchvision.models as models

model = models.googlenet(pretrained=True)
# p.element_size() 取得 p 的 element 占多少 byte
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print("Total memory for parameters: ", param_size)