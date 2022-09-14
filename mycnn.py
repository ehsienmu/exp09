PRINT_SHAPE = False
import torch.nn as nn
class CNN(nn.Module):

    def __init__(self, num_out=10):
        super(CNN, self).__init__()
        # input_shape = (1, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0), # output_shape=(16, 24, 24)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        # input_shape = (16, 24, 24)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0), # output_shape=(32, 22, 22)
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0), # output_shape=(64, 20, 20)
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential( # output_shape=(128, 18, 18)
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv5 = nn.Sequential( # output_shape=(256, 16, 16)
            nn.Conv2d(128, 256, 3, 1, 0),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(256 * 20 * 20, num_out)
        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

    def forward(self, x):
        if PRINT_SHAPE:
            print('input: ', x.shape)
        x = self.conv1(x)
        if PRINT_SHAPE:
            print('after conv1: ', x.shape)
        x = self.conv2(x)
        if PRINT_SHAPE:
            print('after conv2: ', x.shape)
        x = self.conv3(x)
        if PRINT_SHAPE:
            print('after conv3: ', x.shape)
        x = self.conv4(x)
        if PRINT_SHAPE:
            print('after conv4: ', x.shape)
        x = self.conv5(x)
        if PRINT_SHAPE:
            print('after conv5: ', x.shape)
        # flatten the output of conv to (batch_size, 512 * 8 * 8)
        x = x.view(x.size(0), -1)       
        output = self.fc(x)
        return output#, x    # return x for visualization
    
    def lout(self, x, l):
        for layer in self.conv_layers[:l]:
            # print(layer)
            # print(x.shape)
            x = layer(x)
            # print(x.shape)
        return x