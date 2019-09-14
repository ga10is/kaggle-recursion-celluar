import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .. import config
from .loss import ArcMarginProduct


class ResNet(nn.Module):
    def __init__(self, dropout_rate, latent_dim, temperature, m):
        super(ResNet, self).__init__()
        # self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torchvision.models.resnet34(pretrained=True)
        # self.resnet = torchvision.models.resnet18(pretrained=True)
        n_out_channels = 512  # resnet18, 34: 512, resnet50: 512*4

        conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(in_channels=6,
                                      out_channels=conv1.out_channels,
                                      kernel_size=conv1.kernel_size,
                                      stride=conv1.stride,
                                      padding=conv1.padding,
                                      bias=conv1.bias)

        # copy pretrained weights
        self.resnet.conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.resnet.conv1.weight.data[:, 3:, :, :] = conv1.weight.data

        '''
        trained_kernel = self.resnet.conv1.weight
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)
        self.resnet.conv1 = new_conv
        '''

        # FC
        self.norm1 = nn.BatchNorm1d(n_out_channels)
        self.drop1 = nn.Dropout(dropout_rate)
        # FC
        self.fc = nn.Linear(n_out_channels, latent_dim)
        self.norm2 = nn.BatchNorm1d(latent_dim)
        self.arc = ArcMarginProduct(
            latent_dim, config.N_CLASSES, s=temperature, m=m, easy_margin=False)

    def forward(self, x, label=None):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # GAP
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.drop1(x)
        # FC
        x = self.fc(x)
        # x = self.norm2(x)
        # Arc
        x = self.arc(x, label)

        return x


class SEResNet(nn.Module):
    def __init__(self, dropout_rate, latent_dim, temperature, m):
        super(SEResNet, self).__init__()

        senet = pretrainedmodels.__dict__['se_resnext50_32x4d'](
            num_classes=1000, pretrained='imagenet')
        self.layer0 = senet.layer0
        self.layer1 = senet.layer1
        self.layer2 = senet.layer2
        self.layer3 = senet.layer3
        self.layer4 = senet.layer4

        self.norm1 = nn.BatchNorm1d(512 * 4)
        self.drop1 = nn.Dropout(dropout_rate)
        # FC
        self.fc = nn.Linear(512 * 4, latent_dim)
        # self.norm2 = nn.BatchNorm1d(output_neurons)
        self.arc = ArcMarginProduct(
            latent_dim, 2, s=temperature, m=m, easy_margin=False)

    def forward(self, x, label=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # GAP
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.drop1(x)
        # FC
        x = self.fc(x)
        # x = self.norm2(x)
        # Arc
        x = self.arc(x, label)

        return x
