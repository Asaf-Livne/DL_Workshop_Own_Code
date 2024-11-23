## Create an audio discriminator model with a convolutional neural network
## The model is used to classify audio samples as real or fake
## 7 layers of 1D convolutional layers with batch normalization and leaky ReLU activation functions
## 1 fully connected layer with sigmoid activation function
## The model is trained using the binary cross-entropy loss function
## Use nn.sequential to define the model

from imports import *

class Discriminator(nn.Module):
    def __init__(self, kernel, stride, linear_size):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            ConvLayer(1, 256, kernel, stride, 1),
            nn.MaxPool2d(2, 2),
            ConvLayer(256, 512, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvLayer(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvLayer(512, 512, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.Linear(linear_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    

    def forward(self, x):
        return self.seq(x)
    

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.layer(x)

