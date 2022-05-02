import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.pretrained import EncoderClassifier

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
        grad_input = -1*grad_input
        return grad_input

    def grad_reverse(input):
        return GradReverse.apply(input)

class TDNNSexClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TDNNSexClassifier, self).__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.norm = nn.BatchNorm1d(128)
        self.stats_pooling = StatisticsPooling()

        self.classify = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )

    def forward(self, input):
        input = GradReverse.grad_reverse(input)
        input = self.norm(input)
        input = self.tdnn(input)
        input = input.reshape(input.shape[0], input.shape[2], input.shape[1])

        ## statistics pooling ##
        stat_pooling = self.stats_pooling(input)
        stat_pooling = stat_pooling.squeeze(1)

        logits = self.classify(stat_pooling)
        logits = F.log_softmax(logits, 1)
        return logits


class SexClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SexClassifier, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.norm = nn.BatchNorm1d(128)

        self.classify = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.stats_pooling = StatisticsPooling()

    def forward(self, input):
        input = GradReverse.grad_reverse(input)
        input = self.norm(input)
        input = input.reshape(input.shape[0], input.shape[2], input.shape[1])
        logits = self.initial(input)

        ## statistics pooling ##
        stat_pooling = self.stats_pooling(logits)
        stat_pooling = stat_pooling.squeeze(1)

        logits = self.classify(stat_pooling)
        logits = F.log_softmax(logits, 1)
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
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        ## encoder layers ##
        self.encoder=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=1, padding=7),
            GLU(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(num_features=64, affine=True),
            GLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(num_features=64, affine=True),
            GLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(num_features=128, affine=True),
            GLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm1d(num_features=128, affine=True),
            GLU(),
        )

        ## decoder layers ##
        self.decoder=nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            #PixelShuffle(upscale_factor=2),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(num_features=64, affine=True),
            GLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            #PixelShuffle(upscale_factor=2),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.InstanceNorm1d(num_features=32, affine=True),
            GLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=15, stride=1, padding=7)
        )

        ## Sex classifier: num_classes = 2 ##
        self.sex_classifier = TDNNSexClassifier(2)


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

        ## sex classifier ##
        sex_classifier_logits = self.sex_classifier(input)
        #sex_classifier_logits = torch.rand((out.shape[0],2)).to(torch.device("cuda"))

        ## decode ##
        input = self.decoder(input)
        input = input.squeeze(1)
        input = input.reshape(input.shape[0], out.shape[1], out.shape[2])
        ## return reconstructed speech feature for reconstruction loss, sex classification for cross entropy loss ##
        return input, sex_classifier_logits


##########################################################################################
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        # self.residualLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
        #                                              out_channels=out_channels,
        #                                              kernel_size=kernel_size,
        #                                              stride=1,
        #                                              padding=padding),
        #                                    nn.InstanceNorm1d(
        #                                        num_features=out_channels,
        #                                        affine=True),
        #                                    GLU(),
        #                                    nn.Conv1d(in_channels=out_channels,
        #                                              out_channels=in_channels,
        #                                              kernel_size=kernel_size,
        #                                              stride=1,
        #                                              padding=padding),
        #                                    nn.InstanceNorm1d(
        #                                        num_features=in_channels,
        #                                        affine=True)
        #                                    )

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, input):
        h1_norm = self.conv1d_layer(input)
        h1_gates_norm = self.conv_layer_gates(input)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        return input + h2_norm


##########################################################################################
class downSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))

    def forward(self, input):
        # GLU
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


##########################################################################################


class CycleGANGenerator(nn.Module):
    def __init__(self):
        super(CycleGANGenerator, self).__init__()

        ## Sex classifier: num_classes = 2 ##
        self.sex_classifier = SexClassifier(2)
        self.stats_pooling = StatisticsPooling()

        # 2D Conv Layer 
        self.conv1 = nn.Conv2d(in_channels=1,  # TODO 1 ?
                               out_channels=128,
                               kernel_size=(5, 15),
                               stride=(1, 1),
                               padding=(2, 7))

        self.conv1_gates = nn.Conv2d(in_channels=1,  # TODO 1 ?
                                     out_channels=128,
                                     kernel_size=(5, 15),
                                     stride=1,
                                     padding=(2, 7))

        # 2D Downsample Layer
        self.downSample1 = downSample_Generator(in_channels=128,
                                                out_channels=256,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)

        self.downSample2 = downSample_Generator(in_channels=256,
                                                out_channels=256,
                                                kernel_size=5,
                                                stride=2,
                                                padding=2)

        # 2D -> 1D Conv
        # self.conv2dto1dLayer = nn.Sequential(nn.Conv1d(in_channels=2304,
        #                                                out_channels=256,
        #                                                kernel_size=1,
        #                                                stride=1,
        #                                                padding=0),
        #                                      nn.InstanceNorm1d(num_features=256,
        #                                                        affine=True))

        # Residual Blocks
        # self.residualLayer1 = ResidualLayer(in_channels=256,
        #                                     out_channels=512,
        #                                     kernel_size=3,
        #                                     stride=1,
        #                                     padding=1)
        # self.residualLayer2 = ResidualLayer(in_channels=256,
        #                                     out_channels=512,
        #                                     kernel_size=3,
        #                                     stride=1,
        #                                     padding=1)
        # self.residualLayer3 = ResidualLayer(in_channels=256,
        #                                     out_channels=512,
        #                                     kernel_size=3,
        #                                     stride=1,
        #                                     padding=1)
        # self.residualLayer4 = ResidualLayer(in_channels=256,
        #                                     out_channels=512,
        #                                     kernel_size=3,
        #                                     stride=1,
        #                                     padding=1)
        # self.residualLayer5 = ResidualLayer(in_channels=256,
        #                                     out_channels=512,
        #                                     kernel_size=3,
        #                                     stride=1,
        #                                     padding=1)
        # self.residualLayer6 = ResidualLayer(in_channels=256,
        #                                     out_channels=512,
        #                                     kernel_size=3,
        #                                     stride=1,
        #                                     padding=1)

        # 1D -> 2D Conv
        # self.conv1dto2dLayer = nn.Sequential(nn.Conv1d(in_channels=256,
        #                                                out_channels=2304,
        #                                                kernel_size=1,
        #                                                stride=1,
        #                                                padding=0),
        #                                      nn.InstanceNorm1d(num_features=2304,
        #                                                        affine=True))

        # UpSample Layer
        self.upSample1 = self.upSample(in_channels=256,
                                       out_channels=1024,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.upSample2 = self.upSample(in_channels=256,
                                       out_channels=512,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.lastConvLayer = nn.Conv2d(in_channels=128,
                                       out_channels=1,
                                       kernel_size=(5, 15),
                                       stride=(1, 1),
                                       padding=(2, 7))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(
                                           num_features=out_channels,
                                           affine=True),
                                       GLU())

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm2d(
                                           num_features=out_channels // 4,
                                           affine=True),
                                       GLU())
        return self.convLayer


    def forward(self, input):
        # GLU
        input = input.view(input.shape[0], input.shape[2], input.shape[1])
        #print("Generator forward input: ", input.shape)
        input = input.unsqueeze(1)
        #print("Generator forward input: ", input.shape)
        conv1 = self.conv1(input) * torch.sigmoid(self.conv1_gates(input))
        #print("Generator forward conv1: ", conv1.shape)

        # DownloadSample
        downsample1 = self.downSample1(conv1)
        #print("Generator forward downsample1: ", downsample1.shape)
        downsample2 = self.downSample2(downsample1)

        temp = downsample2.view(downsample2.shape[0], downsample2.shape[1], downsample2.shape[3], downsample2.shape[2])
        sex_classifier_input = temp.view(temp.shape[0], temp.shape[1]*temp.shape[2], temp.shape[3])
    
        # mean = torch.mean(sex_classifier_input, 2)
        # std = torch.std(sex_classifier_input, 2)
        # stat_pooling = torch.cat((mean, std), 1)

        stat_pooling = self.stats_pooling(sex_classifier_input)
        stat_pooling = stat_pooling.squeeze(1)

        sex_classifier_logits = self.sex_classifier(stat_pooling)

        #sex_classification_input = downsample2.view(downsample2.shape[0], )
        #sex_classifier_logits = torch.rand((input.shape[0],2)).to(torch.device("cuda"))

        # 2D -> 1D
        # reshape
        # reshape2dto1d = downsample2.view(downsample2.size(0), 2304, 1, -1)
        # reshape2dto1d = reshape2dto1d.squeeze(2)
        # #print("Generator forward reshape2dto1d: ", reshape2dto1d.shape)
        # conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        # #print("Generator forward conv2dto1d_layer: ", conv2dto1d_layer.shape)

        # #residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        # # residual_layer_2 = self.residualLayer2(residual_layer_1)
        # # residual_layer_3 = self.residualLayer3(residual_layer_2)
        # # residual_layer_4 = self.residualLayer4(residual_layer_3)
        # # residual_layer_5 = self.residualLayer5(residual_layer_4)
        # # residual_layer_6 = self.residualLayer6(residual_layer_5)

        # #print("Generator forward residual_layer_6: ", residual_layer_6.shape)

        # # 1D -> 2D
        # conv1dto2d_layer = self.conv1dto2dLayer(conv2dto1d_layer)
        # #print("Generator forward conv1dto2d_layer: ", conv1dto2d_layer.shape)
        # # reshape
        # reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        # reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 20, -1)
        #print("Generator forward reshape1dto2d: ", reshape1dto2d.shape)

        # UpSample
        upsample_layer_1 = self.upSample1(downsample2)#reshape1dto2d)
        #print("Generator forward upsample_layer_1: ", upsample_layer_1.shape)
        upsample_layer_2 = self.upSample2(upsample_layer_1)
        #print("Generator forward upsample_layer_2: ", upsample_layer_2.shape)

        output = self.lastConvLayer(upsample_layer_2)
        #print("Generator forward output: ", output.shape)
        output = output.squeeze(1)
        #print("Generator forward output: ", output.shape)
        output = output.view(output.shape[0], output.shape[2], output.shape[1])
        
        return output, sex_classifier_logits