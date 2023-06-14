# Import Necessary modules
import torch
import torch.nn as nn

# Resnet class
class ResNet(nn.Module):

  '''
  Returns the desired resnet architecture.


  Args:
  num_layers : Number of layers in ResNet
  block : residual block created using Block class inherited from model_builder
  image_channels : Input image channels
  num_classes : Number of image classes we need to classify.


  Returns:
  Desired resnet architecture
  '''

  # Constructor
  def __init__(self,
               num_layers,
               block,
               image_channels,
               num_classes):
    
    # Mking assertsion so that the valid architecture is selected
    assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                  f'to be 18, 34, 50, 101, or 152 '


    super(ResNet, self).__init__()
    if num_layers < 50:
        self.expansion = 1
    else:
        self.expansion = 4
    if num_layers == 18:
        layers = [2, 2, 2, 2]
    elif num_layers == 34 or num_layers == 50:
        layers = [3, 4, 6, 3]
    elif num_layers == 101:
        layers = [3, 4, 23, 3]
    else:
        layers = [3, 8, 36, 3]
    self.in_channels = 64
    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # ResNetLayers
    self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
    self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
    self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
    self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * self.expansion, num_classes)
  
  # Forward function
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x


  def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
    layers = []

    identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                        nn.BatchNorm2d(intermediate_channels*self.expansion))
    layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
    self.in_channels = intermediate_channels * self.expansion # 256
    for i in range(num_residual_blocks - 1):
        layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
    return nn.Sequential(*layers)
