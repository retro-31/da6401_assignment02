import torch.nn as nn

class FlexibleCNN(nn.Module):
    def __init__(self, num_filters=32, activation=nn.ReLU, 
                 filter_organisation='same', use_batchnorm=False, 
                 dropout_rate=0.0, num_classes=10):
        super().__init__()
        layers = []
        in_channels = 3

        filter_sizes = []
        if filter_organisation == 'same':
            filter_sizes = [num_filters] * 5
        elif filter_organisation == 'double':
            filter_sizes = [num_filters * (2 ** i) for i in range(5)]
        elif filter_organisation == 'half':
            filter_sizes = [max(num_filters // (2 ** i), 8) for i in range(5)]

        for out_channels in filter_sizes:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation())
            layers.append(nn.MaxPool2d(2))
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * 7 * 7, num_classes)  # Assuming input images 224x224 and 5 maxpools

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc(x)

