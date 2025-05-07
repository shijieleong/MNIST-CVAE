import torch
import numpy as np
import matplotlib.pyplot as plt

def save_checkpoint(net):
    torch.save(net.state_dict(), './models/trained_model.pth')
    print("Checkpoint saved!")


def load_checkpoint(net, checkpoint_path):
    net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print("Checkpoint Loaded!")
    return net


def show_images(reconstructed, label):
    # Ensure tensors are on CPU and detached
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu()

    # Flattened? Reshape to (batch, 28, 28)
    if reconstructed.ndim == 2 and reconstructed.shape[1] == 784:
        reconstructed = reconstructed.view(-1, 28, 28)

    # Has channel dim? Remove it: (B, 1, 28, 28) -> (B, 28, 28)
    if reconstructed.ndim == 4 and reconstructed.shape[1] == 1:
        reconstructed = reconstructed.squeeze(1)

    # Pick the first image
    for img in reconstructed:
        reconstructed_img = img.numpy()

        # Plot
        fig, axes = plt.subplots(1, 1)

        axes.imshow(reconstructed_img, cmap='gray')
        axes.set_title(f"Reconstructed with label {label}")
        axes.axis('off')

        plt.tight_layout()
        plt.show()