import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# [ BATCH_SIZE x NUM_TIMESTAMPS x MFCC_FEATURE_DIM ]
# need to swap to
# [ BATCH_SIZE x MFCC_FEATURE_DIM x NUM_TIMESTAMPS ]

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input

    def grad_reverse(input):
        return GradReverse.apply(input)

class SexClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SexClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, input):
        input = GradReverse.grad_reverse(input)
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)
        return logits

# Gated Linear Units
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, mfcc_feature_dim):
        super(ConvAutoencoder, self).__init__()
        
        ## model parameters ##
        self.mfcc_feature_dim = mfcc_feature_dim

        ## encoder layers ##
        self.encoder=nn.Sequential(
            nn.Conv1d(in_channels=self.mfcc_feature_dim, out_channels=128, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            #nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            #nn.InstanceNorm1d(num_features=128, affine=True),
            #GLU(),
            #nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            #nn.InstanceNorm1d(num_features=128, affine=True),
            #GLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(num_features=256, affine=True),
            nn.ReLU(),
            #nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            #nn.InstanceNorm1d(num_features=256, affine=True),
            #GLU(),
            #nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            #nn.InstanceNorm1d(num_features=256, affine=True),
            #GLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(num_features=512, affine=True),
            nn.ReLU(),
            #nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2),
            #nn.InstanceNorm1d(num_features=512, affine=True),
            #GLU(),
            # nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2),
            # nn.InstanceNorm1d(num_features=512, affine=True),
            # GLU()
        )

        ## decoder layers ##
        self.decoder=nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2),
            #nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            #nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, stride=1, padding=2),
            #PixelShuffle(upscale_factor=2), 
            nn.ConvTranspose1d(in_channels=1024, out_channels=1024 // 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(num_features=1024 // 2, affine=True),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024 // 2, out_channels=512, kernel_size=5, stride=1, padding=2),
            #nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2),
            #nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2),
            #PixelShuffle(upscale_factor=2), 
            nn.ConvTranspose1d(in_channels=512, out_channels=512 // 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(num_features=512 // 2, affine=True),
            nn.ReLU(),
            #nn.Conv1d(in_channels=512 // 2, out_channels=512 // 2, kernel_size=15, stride=1, padding=7),
            #nn.Conv1d(in_channels=512 // 2, out_channels=512 // 2, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(in_channels=512 // 2, out_channels=self.mfcc_feature_dim, kernel_size=15, stride=1, padding=7)
        )

        ## Sex classifier: num_classes = 2 ##
        self.sex_classifier = SexClassifier(2)


    def forward(self, input):
        ## encode ##
        out = input
        #print(input.shape)
        input = self.encoder(input)
        #print(input.shape)

        ## statistics pooling ##
        mean = torch.mean(input, 2)
        std = torch.std(input, 2)
        stat_pooling = torch.cat((mean, std), 1)

        ## sex classifier ##
        sex_classifier_logits = self.sex_classifier(stat_pooling)
        
        ## decode ##
        input = self.decoder(input)
        #print(input.shape)
        ## return reconstructed speech feature for reconstruction loss, sex classification for cross entropy loss ##
        return input, sex_classifier_logits

class SmallConvAutoencoder(nn.Module):
    def __init__(self, mfcc_feature_dim):
        super(SmallConvAutoencoder, self).__init__()
        
        ## model parameters ##
        self.mfcc_feature_dim = mfcc_feature_dim

        ## encoder layers ##
        self.encoder=nn.Sequential(
            nn.Conv1d(in_channels=self.mfcc_feature_dim, out_channels=160, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(num_features=160, affine=True),
            nn.ReLU()
        )

        ## decoder layers ##
        self.decoder=nn.Sequential(
            nn.ConvTranspose1d(in_channels=160, out_channels=self.mfcc_feature_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(num_features=self.mfcc_feature_dim, affine=True),
            nn.ReLU(),
        )

        ## Sex classifier: num_classes = 2 ##
        #self.sex_classifier = SexClassifier(2)


    def forward(self, input):
        ## encode ##
        out = input
        #print(input.shape)
        input = self.encoder(input)
        #print(input.shape)

        ## statistics pooling ##
        mean = torch.mean(input, 2)
        std = torch.std(input, 2)
        stat_pooling = torch.cat((mean, std), 1)

        ## sex classifier ##
        #sex_classifier_logits = self.sex_classifier(stat_pooling)
        sex_classifier_logits = torch.rand((1,2)).to(torch.device("cuda"))
        
        ## decode ##
        input = self.decoder(input)
        #print(input.shape)
        ## return reconstructed speech feature for reconstruction loss, sex classification for cross entropy loss ##
        return input, sex_classifier_logits

# sanity check model

class FullyConnSexClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FullyConnSexClassifier, self).__init__()
        self.fc1 = nn.Linear(40, 20)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, input):
        input = GradReverse.grad_reverse(input)
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)
        return logits

class FullyConnectedAutoencoder(nn.Module):
    def __init__(self, mfcc_feature_dim, batch_size):
        super(FullyConnectedAutoencoder, self).__init__()
        
        ## model parameters ##
        self.mfcc_feature_dim = mfcc_feature_dim
        self.batch_size = batch_size

        ## encoder layers ##
        self.encoder=nn.Sequential(
            nn.Linear(in_features=self.mfcc_feature_dim, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=20)
        )

        ## decoder layers ##
        self.decoder=nn.Sequential(
            nn.Linear(in_features=20, out_features=40),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=80)
        )

        ## Sex classifier: num_classes = 2 ##
        self.sex_classifier = FullyConnSexClassifier(2)

    def forward(self, input):
        ## encode ##
        out = input
        input = self.encoder(input)

        ## statistics pooling ##
        mean = torch.mean(input.reshape(input.shape[0], input.shape[2], input.shape[1]), 2)
        std = torch.std(input.reshape(input.shape[0], input.shape[2], input.shape[1]), 2)
        stat_pooling = torch.cat((mean, std), 1)

        ## sex classifier ##
        sex_classifier_logits = self.sex_classifier(stat_pooling)
        #sex_classifier_logits = torch.rand((1,2)).to(torch.device("cuda"))
        
        ## decode ##
        input = self.decoder(input)

        ## return reconstructed speech feature for reconstruction loss, sex classification for cross entropy loss ##
        return input, sex_classifier_logits