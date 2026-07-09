import torch
from torch import nn
import torch.nn.functional as F

from models.softmaskLayer import SoftMaskedLayer



class VGG16(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, input_size=32):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, padding=1)
        conv1_outsize = (input_size + 2 * 1 - 3) // 1 + 1
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        conv2_outsize = (conv1_outsize + 2 * 1 - 3) // 1 + 1
        self.pool1 = nn.MaxPool2d(2, 2)
        conv2_outsize = (conv2_outsize - 2 + 2*0) // 2 + 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        conv3_outsize = (conv2_outsize + 2 * 1 - 3) // 1 + 1
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        conv4_outsize = (conv3_outsize + 2 * 1 - 3) // 1 + 1
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        conv4_outsize = (conv4_outsize - 2 + 2*1) // 2 + 1
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        conv5_outsize = (conv4_outsize + 2 * 1 - 3) // 1 + 1
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        conv6_outsize = (conv5_outsize + 2 * 1 - 3) // 1 + 1
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        conv7_outsize = (conv6_outsize + 2 * 1 - 1) // 1 + 1
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        conv7_outsize = (conv7_outsize - 2 + 2*1) // 2 + 1
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        conv8_outsize = (conv7_outsize + 2 * 1 - 3) // 1 + 1
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        conv9_outsize = (conv8_outsize + 2 * 1 - 3) // 1 + 1
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        conv10_outsize = (conv9_outsize + 2 * 1 - 1) // 1 + 1
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        conv10_outsize = (conv10_outsize - 2 + 2*1) // 2 + 1
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        conv11_outsize = (conv10_outsize + 2 * 1 - 3) // 1 + 1
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        conv12_outsize = (conv11_outsize + 2 * 1 - 3) // 1 + 1
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        conv13_outsize = (conv12_outsize + 2 * 1 - 1) // 1 + 1
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self._conv13_outsize = (conv13_outsize - 2 + 2*1) // 2 + 1
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        # self.fc14 = nn.Linear(512*4*4, 1024)
        self.fc14 = nn.Linear(512 * int(
            self._conv13_outsize) ** 2, 1024)
        self.drop1 = nn.Dropout()
        self.fc15 = nn.Linear(1024, 128)
        self.drop2 = nn.Dropout()
        self.fc16 = nn.Linear(128, num_classes)

        self.weight_keys = [['conv1.weight', 'conv1.bias'], ['conv2.weight', 'conv2.bias'], ['conv3.weight', 'conv3.bias'], ['conv4.weight', 'conv4.bias'], ['conv5.weight', 'conv5.bias'], ['conv6.weight', 'conv6.bias'], ['conv7.weight', 'conv7.bias'], ['conv8.weight', 'conv8.bias'], ['conv9.weight', 'conv9.bias'], ['conv10.weight', 'conv10.bias'], ['conv11.weight', 'conv11.bias'], ['conv12.weight', 'conv12.bias'], ['conv13.weight', 'conv13.bias'],
                            ['fc14.weight', 'fc14.bias'],
                            ['fc15.weight', 'fc15.bias'],
                            ['bn1.weight', 'bn1.bias'],
                            ['bn2.weight', 'bn2.bias'], ['bn3.weight', 'bn3.bias'], [
                                'bn4.weight', 'bn4.bias'], ['bn5.weight', 'bn5.bias'],
                            ['fc16.weight', 'fc16.bias']
                            ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        # x = x.view(-1, 512*4*4)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return F.log_softmax(x, dim=1)

    def forward_feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        # feature = torch.flatten(x, 1)
        feature = x.view(x.size(0), -1)
        x = self.fc16(x)

        return F.log_softmax(x, dim=1), feature

def compute_output_size(input_size, kernel_size, padding, stride):
    return (input_size + 2 * padding - kernel_size) // stride + 1

class VGG16softmasking(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, num_tasks=2, input_size=32):
        super(VGG16softmasking, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1 = SoftMaskedLayer(
            "conv2d", num_channels, 64, 3, padding=1)
        conv1_outsize = (input_size + 2 * 1 - 3) // 1 + 1
        # self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = SoftMaskedLayer(
            "conv2d", 64, 64, 3, padding=1)
        conv2_outsize = (conv1_outsize + 2 * 1 - 3) // 1 + 1
        self.pool1 = nn.MaxPool2d(2, 2)
        conv2_outsize = (conv2_outsize - 2 + 2*0) // 2 + 1
        self.bn1 = SoftMaskedLayer("bn", 64)
        self.relu1 = nn.ReLU()

        # self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = SoftMaskedLayer(
            "conv2d", 64, 128, 3, padding=1)
        conv3_outsize = (conv2_outsize + 2 * 1 - 3) // 1 + 1
        # self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = SoftMaskedLayer(
            "conv2d", 128, 128, 3, padding=1)
        conv4_outsize = (conv3_outsize + 2 * 1 - 3) // 1 + 1
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        conv4_outsize = (conv4_outsize - 2 + 2*1) // 2 + 1
        self.bn2 = SoftMaskedLayer("bn", 128)
        self.relu2 = nn.ReLU()

        self.conv5 = SoftMaskedLayer(
            "conv2d", 128, 128, 3, padding=1)
        conv5_outsize = (conv4_outsize + 2 * 1 - 3) // 1 + 1
        self.conv6 = SoftMaskedLayer(
            "conv2d", 128, 128, 3, padding=1)
        conv6_outsize = (conv5_outsize + 2 * 1 - 3) // 1 + 1
        self.conv7 = SoftMaskedLayer(
            "conv2d", 128, 128, 1, padding=1)
        conv7_outsize = (conv6_outsize + 2 * 1 - 1) // 1 + 1
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        conv7_outsize = (conv7_outsize - 2 + 2*1) // 2 + 1
        self.bn3 = SoftMaskedLayer("bn", 128)
        self.relu3 = nn.ReLU()

        self.conv8 = SoftMaskedLayer(
            "conv2d", 128, 256, 3, padding=1)
        conv8_outsize = (conv7_outsize + 2 * 1 - 3) // 1 + 1
        self.conv9 = SoftMaskedLayer(
            "conv2d", 256, 256, 3, padding=1)
        conv9_outsize = (conv8_outsize + 2 * 1 - 3) // 1 + 1
        self.conv10 = SoftMaskedLayer(
            "conv2d", 256, 256, 1, padding=1)
        conv10_outsize = (conv9_outsize + 2 * 1 - 1) // 1 + 1
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        conv10_outsize = (conv10_outsize - 2 + 2*1) // 2 + 1
        self.bn4 = SoftMaskedLayer("bn", 256)
        self.relu4 = nn.ReLU()

        self.conv11 = SoftMaskedLayer(
            "conv2d", 256, 512, 3, padding=1)
        conv11_outsize = (conv10_outsize + 2 * 1 - 3) // 1 + 1
        self.conv12 = SoftMaskedLayer(
            "conv2d", 512, 512, 3, padding=1)
        conv12_outsize = (conv11_outsize + 2 * 1 - 3) // 1 + 1
        self.conv13 = SoftMaskedLayer(
            "conv2d", 512, 512, 1, padding=1)
        conv13_outsize = (conv12_outsize + 2 * 1 - 1) // 1 + 1
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self._conv13_outsize = (conv13_outsize - 2 + 2*1) // 2 + 1
        self.bn5 = SoftMaskedLayer("bn", 512)
        self.relu5 = nn.ReLU()

        # self.fc14 = nn.Linear(512*4*4, 1024)
        self.fc14 = SoftMaskedLayer("linear", 512 * int(
            self._conv13_outsize) ** 2, 1024, bias=True)
        self.drop1 = nn.Dropout()
        # self.fc15 = nn.Linear(1024, 128)
        self.fc15 = SoftMaskedLayer(
            "linear", 1024, 128, bias=True)
        self.drop2 = nn.Dropout()
        # self.fc16 = nn.Linear(128, 10)
        self.fc16 = SoftMaskedLayer(
            "linear", 128, num_classes, num_tasks=num_tasks, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return F.log_softmax(x, dim=1)

    def forward_feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        # feature = torch.flatten(x, 1)
        feature = x.view(x.size(0), -1)
        x = self.fc16(x)

        return F.log_softmax(x, dim=1), feature
    


    def forward_feature_unflatten(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)

        return x