from prelude import *
from efficientnet_pytorch import EfficientNet


# Use EfficientNet (More tuning needed).
class EfficientNetWrapper(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNetWrapper, self).__init__()
        self.effnet = EfficientNet.from_pretrained("efficientnet-b0")
        self.effnet._fc = nn.Linear(self.effnet._fc.in_features, n_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x
        # return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    n_epoch = 7
    batch_size = 4
    n_classes = 5
    device = pick_device(42)
    train_loader = load("./flower_images/training", batch_size)
    val_loader = load("./flower_images/validation", batch_size)
    model = EfficientNetWrapper(n_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
    save(model, "effnet_trained_weights.pt")
