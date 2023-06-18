from prelude import *
import custom_network
import effnet

if __name__ == "__main__":
    batch_size = 4
    n_classes = 5
    device = pick_device()  # Seeds should not matter for eval.
    val_loader = load("./flower_images/validation", batch_size)
    model = custom_network.CustomNetwork(n_classes)
    model = load_weights(model, "trained_weights.pt")
    model.to(device)
    y_true, y_pred = evaluate(model, val_loader, device)
    metrics = compute_metrics(y_true, y_pred, n_classes)
    print_metrics(metrics, n_classes)
