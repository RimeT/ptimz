import torch

from ptimz.model_zoo.unet import resunet50_2d

in_channels = 4
num_classes = 10

model = resunet50_2d(pretrained=False, in_chans=in_channels, num_classes=num_classes)

input = torch.rand((1, in_channels, 224, 224))
with torch.no_grad():
    output = model(input)
print(output.shape)
