from prelude import *


# Custom pure CNN.
class CustomNetwork(nn.Module):
    def __init__(self, n_classes):
        super(CustomNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    n_epoch = 4
    batch_size = 4
    n_classes = 5
    device = pick_device(42)
    train_loader = load("./flower_images/training", batch_size)
    val_loader = load("./flower_images/validation", batch_size)
    model = CustomNetwork(n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.00015)
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
    save(model, "trained_weights.pt")
