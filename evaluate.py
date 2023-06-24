from prelude import *
import custom_network
import effnet
import resnet
import time

if __name__ == "__main__":
    batch_size = 10
    n_classes = 5

    load_start = time.time()
    device = pick_device()  # Seeds should not matter for eval.
    val_loader = load("./flower_images/validation", batch_size)
    model = resnet.resnet18(n_classes)
    model = load_weights(model, "resnet_trained_weights_f32.pt")
    model.to(device)
    load_time = time.time() - load_start
    print(f"Load time: {load_time * 1000:.2f} ms")

    eval_start = time.time()
    y_true, y_pred = evaluate(model, val_loader, device)
    eval_time = time.time() - eval_start

    metrics = compute_metrics(y_true, y_pred, n_classes)
    print_metrics(metrics, n_classes)
    n_image = len(val_loader) * batch_size
    print(
        f"Eval time: {eval_time * 1000:.2f} ms ({float(n_image) / eval_time:.2f} image/s)"
    )
