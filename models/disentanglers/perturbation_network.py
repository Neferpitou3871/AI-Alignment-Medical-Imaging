import torch
import torch.nn as nn
import torch.nn.functional as F

class Perturbator(nn.Module):
    def __init__(self):
        super(Perturbator, self).__init__()
        
        self.enc_block1 = self.conv_block(8, 16)
        self.enc_block2 = self.conv_block(16, 32)
        self.enc_block3 = self.conv_block(32, 64)
        
        self.up_block1 = self.up_conv_block(64, 32)
        self.up_block2 = self.up_conv_block(32, 16)
        self.up_block3 = self.up_conv_block(16, 8)
        self.final_conv = nn.Conv2d(8, 8, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder (Downsampling path)
        x1 = self.enc_block1(x)  # (16, 16, 16)
        x2 = self.enc_block2(x1) # (8, 8, 32)
        x3 = self.enc_block3(x2) # (4, 4, 64)

        x = self.up_block1(x3)   # (8, 8, 32)
        x = self.up_block2(x)    # 16, 16, 16)
        x = self.up_block3(x)    #  (32, 32, 8)
        x = self.final_conv(x)   #  (32, 32, 8)
        return x