import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.conv_blocks import BasicResBlock

class MINE(nn.Module):
    def __init__(self):
        super(MINE, self).__init__()
        self.conv1 = nn.Conv2d(10, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicResBlock(spatial_dims=2, in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=3, stride=stride, norm_name="batch", act_name="relu")]
        for k in range(1, num_blocks):
            layers.append(BasicResBlock(spatial_dims=2, in_channels=out_channels, out_channels=out_channels, 
                                        kernel_size=3, stride=1, norm_name="batch", act_name="relu"))
        return nn.Sequential(*layers)

    def forward(self, z0, a):
        x = torch.cat((z0, a), dim=1)  #(batch_size,10,32,32)
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer4(x)

        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out.squeeze()
