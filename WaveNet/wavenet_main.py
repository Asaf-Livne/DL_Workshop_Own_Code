import torch.utils
from imports import *
from wavenet_train import *
import dataloader as dl

def main():
    params_dict = {'dilation_repeats': 1, 'dilation_depth': 10, 'num_channels': 32, 'kernel_size': 3, 'epochs_num': 100, 'lr': 0.0001, 'quality': 3}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = dl.PairedWavDataset(f'Data/LQ/{params_dict["quality"]}/train', 'Data/HQ/train', 44100, 1600)
    valid_data = dl.PairedWavDataset(f'Data/LQ/{params_dict["quality"]}/test', 'Data/HQ/test', 2*44100, 100)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=False)
    train_loss, valid_loss = gen_train(train_dl, valid_dl, device, params_dict)
    print(f'Train Loss: {train_loss}, Valid Loss: {valid_loss}')


if __name__ == '__main__':
    main()