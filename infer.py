import torch
from torch.nn import functional as F

import utils
from model import ConditionalVAE
from data_loader import load_data


def main():
    # Config
    INPUT_DIM = 784
    LATENT_DIM = 20
    OUTPUT_DIM = 784
    BATCHSIZE=10

    # Load Data
    _, test_loader = load_data(BATCHSIZE)

    # Initialize vae and load checkpoint
    vae = ConditionalVAE(INPUT_DIM, LATENT_DIM, OUTPUT_DIM)
    vae = utils.load_checkpoint(vae, './models/trained_model.pth')
    vae.eval()
    
    # Generate image of 0 to 9
    pic_num = 10
    sample_data = torch.randn(pic_num, LATENT_DIM)
    sample_labels = torch.arange(10).long()
    sample_labels_onehot = F.one_hot(sample_labels, num_classes=10).float()
    sample = torch.cat([sample_data, sample_labels_onehot], dim=1)

    output = vae.decode(sample)
    utils.show_images(output, sample_labels)


if __name__ == "__main__":
    main()  