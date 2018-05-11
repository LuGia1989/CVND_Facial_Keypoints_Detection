## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), grayscale image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        conv1_ch = 32
        conv2_ch = conv1_ch * 2  # 64
        conv3_ch = conv2_ch * 2  # 128
        conv4_ch = conv3_ch * 2  # 256
        fc1_ch = 4096
        fc2_ch = 1028
        output_ch = 2 * 68
        weight_std = 0.005
        weight_mean = 0.0

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (W - F)/S + 1 = (224 - 5)/1 + 1 = (32, 220, 220)
        self.conv1 = nn.Conv2d(1, conv1_ch, 5)
        self.conv1.weight.data.normal_(std=weight_std, mean=weight_mean)
        
        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers
        # (such as dropout or batch normalization) to avoid over fitting
        # output size = (32, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)

        # output size = (W - F)/S + 1 = (110 - 5)/1 + 1 = (64, 106, 106)
        self.conv2 = nn.Conv2d(conv1_ch, conv2_ch, 5)
        self.conv2.weight.data.normal_(std=weight_std, mean=weight_mean)

        # output size = (64, 53, 53)
        self.pool2 = nn.MaxPool2d(2, 2)

        # output size = (W - F)/S + 1 = (53 - 3)/1 + 1 = (128, 51, 51)
        self.conv3 = nn.Conv2d(conv2_ch, conv3_ch, 3)
        self.conv3.weight.data.normal_(std=weight_std, mean=weight_mean)

        # output size = (128, 25, 25)
        self.pool3 = nn.MaxPool2d(2, 2)

        # output = ( W - F)/S + 1 = (25 - 3)/1 + 1 = (256, 23, 23)
        self.conv4 = nn.Conv2d(conv3_ch, conv4_ch, 3)
        self.conv4.weight.data.normal_(std=weight_std, mean=weight_mean)

        # output size = (256, 11, 11)
        self.pool4 = nn.MaxPool2d(2, 2)

        # output size from 256 * 11 * 11 to 4096
        self.fc1 = nn.Linear(conv4_ch * 11 *11, fc1_ch)
        self.fc1.weight.data.normal_(std=weight_std, mean=weight_mean)

        # drop out with p=0.4
        self.fc1_dropout = nn.Dropout(p=0.4)

        # output size from 4096 to 1028
        self.fc2 = nn.Linear(fc1_ch, fc2_ch)
        self.fc2.weight.data.normal_(std=weight_std, mean=weight_mean)

        # drop out with p=0.4
        self.fc2_dropout = nn.Dropout(p=0.4)

        # output layer 2 * 68
        self.output = nn.Linear(fc2_ch, output_ch)
        self.output.weight.data.normal_(std=weight_std, mean=weight_mean)

    def forward(self, x):
        # TODO: Define the feedforward behavior of this model
        # x is the input image and, as an example, here you may choose to include a pool/conv step:
        # x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # flatten out
        x = x.view(x.size(0), -1)

        # Linear layers
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)

        # final output layer
        x = self.output(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
