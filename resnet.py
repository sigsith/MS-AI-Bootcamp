from prelude import *


# Basic residue block.
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stide=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        left = self.relu(self.bn1(self.conv1(x)))
        left = self.bn2(self.conv2(left))
        out = left + self.identity(x)
        out = self.relu(out)
        return out
