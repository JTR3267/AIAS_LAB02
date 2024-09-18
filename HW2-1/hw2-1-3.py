import torchinfo
import torchvision.models as models

model = models.googlenet(pretrained=True)
torchinfo.summary(model, (3, 224, 224), batch_dim=0, depth=4, col_names=("output_size", "num_params"))