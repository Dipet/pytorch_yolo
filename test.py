from efficientnet_pytorch import EfficientNet
import torch
from tensorboardX import SummaryWriter

device = 'cuda'
input_shape = (1, 3, 416, 416)
model = EfficientNet.from_name('efficientnet-b0').eval().to(device)
dummy = torch.rand(input_shape).to(device)
model.forward = model.extract_features
writer = SummaryWriter()
writer.add_graph(model, dummy)

from torchsummary import summary
summary(model, (3, 416, 416), device=device)
