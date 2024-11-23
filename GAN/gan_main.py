import torch.utils
from imports import *
from gan_train import *
import dataloader as dl

def main():
    params_dict = {'dilation_repeats': 1, 'dilation_depth': 10, 'num_channels': 16, 'kernel_size': 3, 'epochs_num': 100, 'lr': 0.001, 'quality': 3}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = dl.PairedWavDataset(f'Data/LQ/{params_dict["quality"]}/train', 'Data/HQ/train', 44100)
    valid_data = dl.PairedWavDataset(f'Data/LQ/{params_dict["quality"]}/test', 'Data/HQ/test', 44100)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_data, batch_size=16, shuffle=False)
    gen_losses, disc_losses, valid_losses, valid_best_loss = gen_and_disc_train(train_dl, valid_dl, device, params_dict)
    print(f'Train Loss: {gen_losses[-1], disc_losses[-1]}, Valid Loss: {valid_best_loss}')


if __name__ == '__main__':
    main()