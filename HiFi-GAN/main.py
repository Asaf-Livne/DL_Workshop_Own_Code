from imports import *
from train import *

def main():
    cfg = {
        'segment_size': 16384,
        'sample_rate': 44100,
        'n_fft': 4096,
        'win_length': 1024,
        'hop_length': 520,
        'n_mels': 256,
        'center': True,
        'device': 'mps',
        'train_lq_path': 'Data/LQ/24/train/Chunks',
        'train_hq_path': 'Data/HQ Light/train/Chunks',
        'valid_lq_path': 'Data/LQ/24/test/Chunks',
        'valid_hq_path': 'Data/HQ Light/test/Chunks',
        'epochs': 100,
        'lr': 0.001,
        'beta1': 0.8,
        'beta2': 0.99,
        'load': 'Last',
        'batch_size': 16,

        'upsampling_kernels': [16, 16, 8, 4],
        'upsampling_strides': [8, 8, 4, 2],

        'dilations': [[1,3,5], [1,3,5], [1,3,5]],
        'num_channels': 128,
        'mrf_kernels': [3, 7, 11],
        'num_resblocks': 3,
        'num_layers': 3,

        'msd_poolings': [1, 2, 4],
        'mpd_periods': [2, 3, 5, 7, 11]
    }
    train(cfg)


if __name__ == '__main__':
    main()