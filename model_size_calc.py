from models.ConvAutoEncoder import ConvAutoencoder, CycleGANGenerator
from models.FullyConnected import FullyConnectedAutoencoder
from models.EndToEnd import ConvReconstruction

import torch
from torchsummary import summary



model_fcae = FullyConnectedAutoencoder(80, 1).to('cpu')
model_convae = ConvAutoencoder().to('cpu')



summary(model_fcae, (1, 80*10), device='cpu')


summary(model_convae,  (1, 80*10), device='cpu')

model_e2e = ConvReconstruction().to('cpu')





summary(model_e2e,  (1, 80*10), device='cpu')