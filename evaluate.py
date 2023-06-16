import train_and_eval
import torch

if __name__ == "__main__":
    batch_size = 4
    n_classes = 5
    device = torch.device(train_and_eval.select_backend(42))
    val_loader = train_and_eval.load("./flower_images/validation", batch_size)
    model = train_and_eval.CustomNetwork(n_classes)
    model = train_and_eval.load_weights(model, "trained_weights.pt")
    model.to(device)
    y_true, y_pred = train_and_eval.evaluate(model, val_loader, device)
    metrics = train_and_eval.compute_metrics(y_true, y_pred, n_classes)
    train_and_eval.print_metrics(metrics, n_classes)
