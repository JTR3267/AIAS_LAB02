import torchinfo
import torchvision.models as models

model = models.googlenet(pretrained=True)
# depth 取 4，googlenet 有第 4 層 layer
torchinfo.summary(model, (3, 224, 224), batch_dim=0, depth=4, col_names=("input_size", "output_size", "num_params"))