import torch
from torch import nn

class YourFirstNet(torch.nn.Module):
    def __init__(self, n_labels):
        super(YourFirstNet, self).__init__()
        #raise NotImplementedError()
        # TODO: Write your code here

        # 28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 14x14
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 7 x 7
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, padding=1)
        )

        # 4x4

        self.fc1 = nn.Sequential(
            nn.Linear(33856, 256),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(256, n_labels)
    
    def forward(self, X):
        batch_size = X.size(0)
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(batch_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out