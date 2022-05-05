from models.ConvAutoEncoder import ConvAutoencoder, CycleGANGenerator
from models.FullyConnected import DummyFullyConnectedAutoencoder, FullyConnectedAutoencoder
from models.EndToEnd import ConvReconstruction

import torch
from torchsummary import summary


model_convae = ConvAutoencoder().to('cpu')

param_size = 0
for param in model_convae.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model_convae.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model: Convolutional AutoEncoder size: {:.3f}MB'.format(size_all_mb))


summary(model_convae,  (1, 80*10), device='cuda')

model_e2e = ConvReconstruction()

param_size = 0
for param in model_e2e.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model_e2e.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model: Gated Conv AutoEncoder size: {:.3f}MB'.format(size_all_mb))

summary(model_e2e,  (1, 80*10), device='cuda')