import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 28 * 39, out_features=64)  # Adjust in_features according to the output of the last conv layer
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Softmax is not used here as it is included in nn.CrossEntropyLoss during training
        return x


if __name__ == '__main__':
    # You should create an instance of the model and define a loss function and optimizer to use.
    input_x = torch.randn(1, 1, 128, 173)
    model = CNNModel()
    output = model(input_x)
    pdb.set_trace()
    print(f"Output shape: {output.shape}")
