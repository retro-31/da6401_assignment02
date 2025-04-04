import torch.nn as nn

class FlexibleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, num_conv_layers=5,
                 num_filters=32, kernel_size=3, activation=nn.ReLU, dense_neurons=128):
        super().__init__()
        layers = []
        in_channels = input_channels
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2),
                activation(),
                nn.MaxPool2d(2)
            ]
            in_channels = num_filters
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(num_filters * (224 // (2 ** num_conv_layers)) ** 2, dense_neurons),
            activation(),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.dense_layers(x)
