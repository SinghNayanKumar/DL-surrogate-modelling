import torch
import torch.nn as nn
import torch.nn.functional as F

class SE_Block3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D inputs.
    This block learns to re-calibrate channel-wise feature responses by explicitly
    modelling interdependencies between channels. It helps the network focus on
    more informative feature channels.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE_Block3D, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1) # Global Average Pooling
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c) # Squeeze
        y = self.excitation(y).view(b, c, 1, 1, 1) # Excitation
        return x * y.expand_as(x) # Re-calibrate

class DoubleConv3D(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
        
class AttnDoubleConv3D(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2 + SE_Block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se_block = SE_Block3D(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        return self.se_block(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        conv_block = AttnDoubleConv3D if use_attention else DoubleConv3D
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            conv_block(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        conv_block = AttnDoubleConv3D if use_attention else DoubleConv3D
        self.conv = conv_block(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if the size doesn't match x2 due to odd dimensions
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
        
class UNet3D(nn.Module):
    """ Standard 3D U-Net Model """
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(UNet3D, self).__init__()
        conv_block = AttnDoubleConv3D if use_attention else DoubleConv3D
        
        self.inc = conv_block(in_channels, 64)
        self.down1 = Down(64, 128, use_attention)
        self.down2 = Down(128, 256, use_attention)
        self.down3 = Down(256, 512, use_attention)
        self.up1 = Up(512, 256, use_attention)
        self.up2 = Up(256, 128, use_attention)
        self.up3 = Up(128, 64, use_attention)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
    
class UNet3D_Small(nn.Module):
    """ A smaller, faster version of the 3D U-Net with fewer channels. """
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(UNet3D_Small, self).__init__()
        conv_block = AttnDoubleConv3D if use_attention else DoubleConv3D
        
        # Reduced channel sizes from 64->128->256->512 to 32->64->128->256
        self.inc = conv_block(in_channels, 32)
        self.down1 = Down(32, 64, use_attention)
        self.down2 = Down(64, 128, use_attention)
        self.down3 = Down(128, 256, use_attention)
        self.up1 = Up(256, 128, use_attention)
        self.up2 = Up(128, 64, use_attention)
        self.up3 = Up(64, 32, use_attention)
        self.outc = OutConv(32, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits