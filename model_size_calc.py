from models.ConvAutoEncoder import ConvAutoencoder, CycleGANGenerator
from models.FullyConnected import DummyFullyConnectedAutoencoder, FullyConnectedAutoencoder
from models.EndToEnd import ConvReconstruction




model_convae = ConvAutoencoder()

param_size = 0
for param in model_convae.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model_convae.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model: Convolutional AutoEncoder size: {:.3f}MB'.format(size_all_mb))
model_e2e = ConvReconstruction()

