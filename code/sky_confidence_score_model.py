"""
===============================================================================
========================You can run this script directly=======================
===============================================================================

This is the model definition of the sky confidence score model
For more detail about the design, please refer to the report

you can get output like this:
    batch size: 10
    input shape: torch.Size([10, 3, 64, 64])
    output shape: torch.Size([10, 2])

===============================================================================
===============================================================================
"""

import torch
from torch import nn
from torchvision import models
from torchvision.models import DenseNet121_Weights
from torchvision.models.resnet import ResNet50_Weights


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # use the pretrained model here

        self.densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])

        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        dense_features = self.densenet(x)
        # print(dense_features.shape)

        dense_features = dense_features.view(dense_features.size(0), -1)
        # print(dense_features.shape)

        resnet_features = self.resnet(x)
        # print(resnet_features.shape)

        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        # print(resnet_features.shape)

        features = torch.cat((dense_features, resnet_features), dim=1)
        features = self.dropout(features)

        # print(features.shape)

        return features


class FullConnector(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullConnector, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        # x = self.sigmoid(x)
        return x


class Model(nn.Module):
    def __init__(self, feature_extractor, full_connector):
        super(Model, self).__init__()
        self.feature_extractor = feature_extractor
        self.full_connector = full_connector

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.full_connector(x)
        return x


# init the feature extractor models
feature_extractor = FeatureExtractor()

# the essential of this model is a classifier, so we use full connection to reduce the dim to 2
# the cat feature number of densenet and resnet is 6144
full_connector = FullConnector(in_features=6144, out_features=2)

# combine the above two modules
model = Model(feature_extractor, full_connector)

if __name__ == '__main__':
    torch.manual_seed(42)

    batch_size = 10
    channels = 3
    height = 64
    width = 64
    input_tensor = torch.randn(batch_size, channels, height, width)

    model = Model(feature_extractor, full_connector)

    output_tensor = model(input_tensor)

    print(f'batch size: {batch_size}')
    print(f"input shape: {input_tensor.shape}")
    print(f"output shape: {output_tensor.shape}")
