import torchvision.models as models

model = models.googlenet(pretrained=True)
# p.numel() 取得 p 所有 parameter 維度相乘結果
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters: ", total_params)