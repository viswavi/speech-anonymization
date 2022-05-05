import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.pretrained import EncoderClassifier

# [ BATCH_SIZE x NUM_TIMESTAMPS x MFCC_FEATURE_DIM ]
# need to swap to
# [ BATCH_SIZE x MFCC_FEATURE_DIM x NUM_TIMESTAMPS ]

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
class ConvReconstruction(nn.Module):
    def __init__(self):
        super(ConvReconstruction, self).__init__()

        self.encoder=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=1, padding=7),
            nn.InstanceNorm1d(num_features=32, affine=True),
            GLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(num_features=64, affine=True),
            GLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(num_features=64, affine=True),
            GLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(num_features=32, affine=True),
            GLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=15, stride=1, padding=7)
        )

        ## Sex classifier: num_classes = 2 ##
        self.sex_classifier = EncoderClassifier.from_hparams(
            source="/home/ec2-user/capstone/speech-anonymization/results/gender_classifier/1230/save/",
            hparams_file="evaluator_inference.yaml",
            savedir="/home/ec2-user/capstone/speech-anonymization/results/gender_classifier/1230/save/",
        )


    def forward(self, input):
        ## encode ##
        out = input
        input = input.reshape(input.shape[0], input.shape[1]*input.shape[2])

        # batch_size feats timesteps

        # batch_size feats timesteps
        # -->
        # batch_size channel feats*timesteps
        input = input.unsqueeze(1)

        input = self.encoder(input)

        input = input.squeeze(1)
        #print("Generator forward output: ", output.shape)
        input = input.reshape(input.shape[0], out.shape[1], out.shape[2])

        ## sex classifier ##
        sex_classifier_logits, score, index = self.sex_classifier(input)
        #sex_classifier_logits = torch.rand((out.shape[0],2)).to(torch.device("cuda"))

        ## return reconstructed speech feature for reconstruction loss, sex classification for cross entropy loss ##
        return input, sex_classifier_logits


