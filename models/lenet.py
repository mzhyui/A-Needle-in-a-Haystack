import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from models.softmaskLayer import SoftMaskedLayer

class CNNMnist(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, num_classes)

        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def initialize_weights_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data = torch.randn(m.weight.data.size())
                m.bias.data = torch.randn(m.bias.data.size())


class lenet(torch.nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=3),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(inplace=True)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 50, kernel_size=3),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class lenetMini(torch.nn.Module):
    def __init__(self):
        super(lenetMini, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=3),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(inplace=True)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 50, kernel_size=3),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, 128),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LeNet_softmask(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, num_tasks=2, input_size=32):
        super(LeNet_softmask, self).__init__()
        self.conv1 = SoftMaskedLayer(
            "conv2d", num_channels, 6, 5, num_tasks=num_tasks, padding=1)
        conv1_outsize = (input_size + 2 * 1 - 5) / 1 + 1
        conv1_outsize //= 2
        self.conv2 = SoftMaskedLayer(
            "conv2d", 6, 16, 5, num_tasks=num_tasks, padding=1)
        conv2_outsize = (conv1_outsize + 2 * 1 - 5) / 1 + 1
        conv2_outsize //= 2
        self.fc3 = SoftMaskedLayer(
            "linear", 16*(int(conv2_outsize) ** 2), 120, num_tasks=num_tasks, bias=True)
        self.fc4 = SoftMaskedLayer(
            "linear", 120, 84, num_tasks=num_tasks, bias=True)
        self.fc5 = SoftMaskedLayer(
            "linear", 84, num_classes, num_tasks=num_tasks, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out

    def forward_feature(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        feature = torch.flatten(out, 1)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out, feature
    
    def forward_feature_unflatten(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        feature = out
        return feature



class LeNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=10, input_size=32):
        super(LeNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5, padding=2)
        conv1_outsize = (input_size + 2 * 1 - 5) / 1 + 1
        conv1_outsize //= 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        conv2_outsize = (conv1_outsize + 2 * 1 - 5) / 1 + 1
        conv2_outsize //= 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * (int(conv2_outsize) ** 2), 120)  # Assuming 32x32 input -> 5x5 after convs and pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # First conv block: conv -> relu -> maxpool
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        
        # Second conv block: conv -> relu -> maxpool
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on final layer (handled by loss function)
        
        return x