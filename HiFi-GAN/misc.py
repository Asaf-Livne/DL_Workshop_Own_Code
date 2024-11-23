from imports import *


def pad(kernel, dilation):
    return ((kernel * dilation) - dilation) // 2

def init_weights(m):
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


def wn(n):
    return nn.utils.weight_norm(n)

def lrlu(x):
    return nn.LeakyReLU(0.1)(x)

def stack_input(x, p):
    batch, channels, t = x.size()
    if t % p != 0:
        x_r = F.pad(x, (0, p - (t % p)))
        t += p - (t % p)
        x_r = x_r.view(batch, channels, t // p, p)
        return x_r
    x_r = x.view(batch, channels, t // p, p)
    return x_r

def log_mel_spectrogram(cfg, y):
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=cfg['sample_rate'], n_fft=cfg['n_fft'], win_length=cfg['win_length'], hop_length=cfg['hop_length'], n_mels=cfg['n_mels'], center=cfg['center'])(y)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    return mel

def mel_spectrogram(cfg, y):
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=cfg['sample_rate'], n_fft=cfg['n_fft'], win_length=cfg['win_length'], hop_length=cfg['hop_length'], n_mels=cfg['n_mels'], center=cfg['center'])(y)
    return mel