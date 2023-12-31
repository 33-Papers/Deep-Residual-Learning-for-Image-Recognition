# Import Necessary modules
import torch
import torch.nn as nn


# Create a class for Residual Block
class Block(nn.Module):
  '''
  Creates residual block of the selected Resnet Architecture


  Args:
  num_layers : Number of layers in ResNet
  in_channels : Number of input channels for the convolution
  out_channels : Number of output channels of convolution
  identity_downsample : Set to some downsample if required, default : None
  stride : Stride added before convolution


  Returns:
  Residual blocks in order
  '''

  # Constructor
  def __init__(self,
               num_layers,
               in_channels,
               out_channels,
               identity_downsample = None,
               stride = 1):
    
    # Make an assertsion that the Resnet is valid
    assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
    super(Block, self).__init__()

    self.num_layers = num_layers

    # Resnets with layer larger than 34 have convolution channels x 4 in the last layer of block 
    if self.num_layers > 34:
      self.expansion = 4
    else:
      self.expansion = 1


    # Resnet > Resnet34 needs additional layer of 1x1 kernels
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1, padding = 0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    if self.num_layers > 34:
      self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    else:
      # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
      self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample = identity_downsample

  
  # Forward function
  def forward(self, x):
    identity = x
    if self.num_layers > 34:
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)

    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x
