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
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        left = self.relu(self.bn1(self.conv1(x)))
        left = self.bn2(self.conv2(left))
        out = left + self.identity(x)
        out = self.relu(out)
        return out


def block_chain(in_channel, channel, length, do_down_sampling):
    blocks = []
    stride = 2 if do_down_sampling else 1
    blocks.append(Block(in_channel, channel, stride))
    for _ in range(1, length):
        blocks.append(Block(channel, channel, 1))
    return nn.Sequential(*blocks)


# Generic basic Resnet. No bottleneck blocks.
class ResNetShallow(nn.Module):
    def __init__(self, block_config, n_classes):
        super(ResNetShallow, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = block_chain(64, 64, block_config[0], False)
        self.conv3 = block_chain(64, 128, block_config[1], True)
        self.conv4 = block_chain(128, 256, block_config[2], True)
        self.conv5 = block_chain(256, 512, block_config[3], True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(n_classes):
    return ResNetShallow([2, 2, 2, 2], n_classes)


def resnet34(n_classes):
    return ResNetShallow([3, 4, 6, 3], n_classes)


if __name__ == "__main__":
    n_epoch = 20
    batch_size = 128
    n_classes = 5
    device = pick_device(42)
    train_loader = load("./flower_images/training", batch_size)
    val_loader = load("./flower_images/validation", batch_size)
    model = resnet18(n_classes)
    optimizer = optim.Adam(model.parameters(), 0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    model = train(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        n_epoch,
        n_classes,
        device,
        "results.json",
    )
    save(model, "resnet_trained_weights.pt")