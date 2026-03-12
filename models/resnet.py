import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from models.softmaskLayer import SoftMaskedLayer


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def _random_weights(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.weight.data = torch.randn(m.weight.data.size())
        if m.bias is not None:
            m.bias.data = torch.randn(m.bias.data.size())
    if isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlockSM(nn.Module):
    expansion = 1
    

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlockSM, self).__init__()
        self.conv1 = SoftMaskedLayer("conv2d", 
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = SoftMaskedLayer("bn", planes)
        self.conv2 = SoftMaskedLayer("conv2d", planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = SoftMaskedLayer("bn", planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    SoftMaskedLayer("conv2d", in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    SoftMaskedLayer("bn", self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        mid = self.shortcut(x)
        out += mid
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        mid = self.shortcut(x)
        out += mid
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, in_channels=3, num_classes=10, feature_dims=-1, option='B', rand_init: int = 0):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, option=option)
        self.linear = nn.Linear(
            64, num_classes) if feature_dims == -1 else nn.Linear(64, feature_dims)

        # self.apply(_weights_init) if not rand_init else self.apply(_random_weights)
        if rand_init:
            torch.manual_seed(rand_init)
            self.apply(_random_weights)
        else:
            self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetSM(nn.Module):

    def __init__(self, block, num_blocks, in_channels=3, num_classes=10, feature_dims=-1, option='B', rand_init: int = 0):
        super(ResNetSM, self).__init__()
        self.in_planes = 16

        self.conv1 = SoftMaskedLayer("conv2d",
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = SoftMaskedLayer("bn", 16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, option=option)
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, option=option)
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, option=option)
        self.linear = SoftMaskedLayer("linear", 
            64, num_classes, bias=True) if feature_dims == -1 else SoftMaskedLayer("linear", 64, feature_dims, bias=True)

        # self.apply(_weights_init) if not rand_init else self.apply(_random_weights)
        if rand_init:
            torch.manual_seed(rand_init)
            self.apply(_random_weights)
        else:
            self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def forward_feature(self, x):
        # print(f"input shape: {x.shape}")
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print(f"output shape: {out.shape}")
        feature = out.view(out.size(0), -1)
        # print(f"feature shape: {feature.shape}")
        return out, feature