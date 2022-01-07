import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)        
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        return fc2, [conv1, conv2]


class LinearNet(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(LinearNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 100)
        self.layer3 = nn.Linear(100, 50)
        self.layer4 = nn.Linear(50, 20)
        self.layer5 = nn.Linear(20, output_dim)
        
    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        x4 = F.relu(self.layer4(x3))
        x = self.layer5(x4)
        return x, [x1,x2,x3]