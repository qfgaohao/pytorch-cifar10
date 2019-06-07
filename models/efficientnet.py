import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

def round_channels(num_channels: int, width_coeff: float=1, 
                   divisor: int=8, min_depth=None) -> int:
    """Round number of filters based on width coefficient."""
    if width_coeff == 1:
        return num_channels
    min_depth = min_depth or divisor
    new_num_channels = max(min_depth, int(num_channels * width_coeff + divisor / 2) // divisor * divisor)
    # make sure the round down does not go down by more than 10%
    if new_num_channels < 0.9 * num_channels:
        new_num_channels += divisor
    return int(new_num_channels)


def round_repeats(repeats: int, depth_coeff: float=1) -> int:
    """Round number of repeated layers based on depth_coeff."""
    return int(math.ceil(depth_coeff * repeats))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class DropConnect(nn.Module):
    def __init__(self, drop_connect_ratio: float=0.0):
        super(DropConnect, self).__init__()
        self.drop_connect_rate = drop_connect_ratio
    
    def forward(self, x: torch.Tensor):
        if self.training:
            keep_prob = 1.0 - self.drop_connect_rate
            batch_size = x.size(0)
            random_tensor = keep_prob + torch.rand(batch_size, 1, 1, 1, dtype=x.dtype, device=x.device)
            mask = torch.floor(random_tensor)
            x = torch.div(x, keep_prob) * mask
        return x


class MBConvBlock(nn.Module):
    """Inverted Residual Bootleneck."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int,
                 expand_ratio: int, se_ratio: float,
                 id_skip: bool=True, drop_connect_rate:float=0.2,
                 batch_norm_eps: float=1e-05, batch_norm_momentum: float=0.1
                ):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip and stride == 1 and in_channels == out_channels
        self.use_se = (0 < se_ratio <= 1)
        
        # Use num_channels to track output channels.
        num_channels = in_channels
        if self.expand_ratio > 1:
            num_channels = in_channels * self.expand_ratio
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, num_channels,
                                         kernel_size=1, bias=False),
                nn.BatchNorm2d(num_channels, eps=batch_norm_eps,
                                      momentum=batch_norm_momentum),
                Swish()
            )
                
            
        self.depthwise = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 
                                        kernel_size, stride,
                                        padding=kernel_size // 2,
                                        groups=num_channels,
                                        bias=False),
            nn.BatchNorm2d(num_channels, eps=batch_norm_eps,
                                  momentum=batch_norm_momentum),
            Swish()
        )
        
        if self.use_se:
            num_reduced_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_channels, num_reduced_channels, kernel_size=1, stride=1),
                Swish(),
                nn.Conv2d(num_reduced_channels, num_channels, kernel_size=1, stride=1)
            )
        self.project = nn.Sequential(
            nn.Conv2d(num_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=batch_norm_eps,
                                  momentum=batch_norm_momentum)
        )
        if self.id_skip:
            self.drop_connect = DropConnect(drop_connect_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = x
        if self.expand_ratio > 1:
            x = self.expand(x)
        x = self.depthwise(x)

        if self.use_se:
            y = self.se(x)
            x = torch.sigmoid(y) * x  # TODO: CONVERT THIS TO A MODULE
        
        x = self.project(x)
        
        if self.id_skip:
            x = self.drop_connect(x)
            x += inputs
        return x


class EfficientNet(nn.Module):
    
    def __init__(self, num_classes: int, width_coeff:float, depth_coeff:float,
                 id_skip: bool=True, drop_connect_ratio: float=0.2, dropout_rate:float=0.2,
                 batch_norm_eps: float=1e-03, batch_norm_momentum: float=0.99):
        
        super(EfficientNet, self).__init__()
        
        block_params = [
            #repeats, kernel_size, stride, expand_ratio, in_channels, out_channels, se 
            (1, 3, 1, 1, 32, 16, 0.25),
            (2, 3, 2, 6, 16, 24, 0.25),
            (2, 5, 2, 6, 24, 40, 0.25),
            (3, 3, 2, 6, 40, 80, 0.25),
            (3, 5, 1, 6, 80, 112, 0.25),
            (4, 5, 2, 6, 112, 192, 0.25),
            (1, 3, 1, 6, 192, 320, 0.25)
        ]
        
        # the first conv
        num_channels = round_channels(block_params[0][4], width_coeff)
        stem = nn.Conv2d(3, num_channels, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        bn0 = nn.BatchNorm2d(num_channels, eps=batch_norm_eps,
                                  momentum=batch_norm_momentum)
        swish0 = Swish()
        
        # the blocks
        blocks = []
        for (repeats, kernel_size, stride, expand_ratio, in_channels, out_channels, se_ratio) in block_params:
            in_channels = round_channels(in_channels, width_coeff)
            out_channels = round_channels(out_channels, width_coeff)
            repeats = round_repeats(repeats, depth_coeff)
            
            blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio, id_skip, dropout_rate,
                 batch_norm_eps, batch_norm_momentum))
            if repeats > 1:
                in_channels = out_channels
                stride = 1
            for _ in range(repeats - 1):
                blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio, id_skip, dropout_rate,
                 batch_norm_eps, batch_norm_momentum))
        blocks = [stem, bn0, swish0] + blocks
        self.features = nn.Sequential(*blocks)
        
        # classification head
        in_channels = round_channels(block_params[-1][5], width_coeff)
        out_channels = round_channels(1280, width_coeff)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=batch_norm_eps, momentum=batch_norm_momentum),
            Swish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(dropout_rate),
            Flatten(),
            nn.Linear(out_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def efnet_b0(num_classes):
    return EfficientNet(num_classes, 1.0, 1.0, dropout_rate=0.2)


def efnet_b1(num_classes):
    return EfficientNet(num_classes, 1.0, 1.1, dropout_rate=0.2)


def efnet_b2(num_classes):
    return EfficientNet(num_classes, 1.1, 1.2, dropout_rate=0.3)


def efnet_b3(num_classes):
    return EfficientNet(num_classes, 1.2, 1.4, dropout_rate=0.3)


def efnet_b4(num_classes):
    return EfficientNet(num_classes, 1.4, 1.8, dropout_rate=0.4)


def efnet_b5(num_classes):
    return EfficientNet(num_classes, 1.4, 1.8, dropout_rate=0.4)


def efnet_b6(num_classes):
    return EfficientNet(num_classes, 1.8, 2.6, dropout_rate=0.5)


def efnet_b7(num_classes):
    return EfficientNet(num_classes, 2.0, 3.1, dropout_rate=0.5)

