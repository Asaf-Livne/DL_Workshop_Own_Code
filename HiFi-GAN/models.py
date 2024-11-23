from imports import *
from misc import pad, init_weights, wn, lrlu, stack_input

class ResBlock (nn.Module):
    def __init__(self, dilations, kernel_size, num_channels):
        super(ResBlock, self).__init__()
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.num_layers = len(dilations)
        self.convs = nn.ModuleList()
        for m in range(self.num_layers):
            self.convs.append(wn(nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, dilation=self.dilations[m], padding=pad(self.kernel_size, self.dilations[m]))))
        for conv in self.convs:
            init_weights(conv)


    def forward(self, x):
        res = x
        x_conv = x
        for conv in self.convs:
            x_conv = lrlu(x_conv)
            x_conv = conv(x_conv)
        return res + x_conv
    

class MRF(nn.Module):
    def __init__(self, dilations, kernel_sizes, num_channels):
        super(MRF, self).__init__()
        self.dilations = dilations
        self.kernel_sizes = kernel_sizes
        self.num_channels = num_channels
        self.num_resblocks = len(dilations)
        self.resblocks = nn.ModuleList()
        for n in range(self.num_resblocks):
            self.resblocks.append(ResBlock(self.dilations[n], self.kernel_sizes[n], self.num_channels))
        
    def forward(self, x):
        xs = None
        for resblock in self.resblocks:
            if xs is None:
                xs = resblock(x)
            else:
                xs += resblock(x)
        xs /= self.num_resblocks
        return xs
    
class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.upsampling_kernels = cfg['upsampling_kernels']
        self.upsampling_strides = cfg['upsampling_strides']
        self.dilations = cfg['dilations']
        self.mrf_kernels = cfg['mrf_kernels']
        self.num_channels = cfg['num_channels']
        self.n_mels = cfg['n_mels']

        self.pre_conv = wn(nn.Conv1d(self.n_mels, self.num_channels, 7, padding=3))

        self.upsampling = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsampling_kernels)):
            stride = self.upsampling_strides[i]
            kernel = self.upsampling_kernels[i]
            in_channels = self.num_channels // 2**i
            out_channels = self.num_channels // 2**(i+1)
            self.upsampling.append(wn(nn.ConvTranspose1d(in_channels, out_channels, kernel, stride=stride, padding=(kernel-stride)//2)))
            mrf_kernels = self.mrf_kernels
            dilations = self.dilations
            self.mrfs.append(MRF(dilations, mrf_kernels, out_channels))
        
        self.post_conv = wn(nn.Conv1d(self.num_channels // 2**len(self.upsampling_kernels), 1, 7, padding=3))
        init_weights(self.pre_conv)
        for upsample in self.upsampling:
            init_weights(upsample)
        init_weights(self.post_conv)

    def forward(self, x):
        x = self.pre_conv(x)
        for i in range(len(self.upsampling)):
            x = lrlu(x)
            x = self.upsampling[i](x)
            x = self.mrfs[i](x)
        x = lrlu(x)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x

class PDiscriminator(nn.Module):
    def __init__(self, period, kernel = 5, stride = 3):
        super(PDiscriminator, self).__init__()
        self.period = period
        self.convs = nn.ModuleList()
        self.convs.append(wn(nn.Conv2d(1, 32, kernel_size=(kernel,1), stride=(stride,1), padding = (pad(kernel, stride), 0))))
        self.convs.append(wn(nn.Conv2d(32, 128, (kernel,1), stride=(stride,1), padding = (pad(kernel, stride), 0))))
        self.convs.append(wn(nn.Conv2d(128, 512, (kernel,1), stride=(stride,1), padding = (pad(kernel, stride), 0))))
        self.convs.append(wn(nn.Conv2d(512, 1024, (kernel,1), stride=(stride,1), padding = (pad(kernel, stride), 0))))
        self.convs.append(wn(nn.Conv2d(1024, 1024, (kernel,1), stride=(stride,1), padding = (pad(kernel, stride), 0))))
        self.convs.append(wn(nn.Conv2d(1024, 1, (3,1), padding = (1, 0))))
        for conv in self.convs:
            init_weights(conv)

    def forward(self, x):
        features = []
        x = stack_input(x, self.period)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if i != len(self.convs) - 1:
                x = lrlu(x)
            features.append(x)
        x = torch.flatten(x, 1)
        return x, features

class MPD(nn.Module):
    def __init__(self, periods):
        super(MPD, self).__init__()
        self.periods = periods
        self.pds = nn.ModuleList()
        for period in self.periods:
            self.pds.append(PDiscriminator(period))
        
    def forward(self, real, gen):
        real_features = []
        gen_features = []
        real_preds = []
        gen_preds = []
        for i in range(len(self.periods)):
            real_pred, real_feature = self.pds[i](real)
            gen_pred, gen_feature = self.pds[i](gen)
            real_preds.append(real_pred)
            gen_preds.append(gen_pred)
            real_features.append(real_feature)
            gen_features.append(gen_feature)
        return real_preds, gen_preds, real_features, gen_features


class SDiscriminator(nn.Module):
    def __init__ (self, pooling):
        super(SDiscriminator, self).__init__()
        self.pooling = nn.AvgPool1d(kernel_size=pooling, stride=pooling // 2) if pooling != 1 else None
        self.convs = nn.ModuleList()
        self.convs.append(wn(nn.Conv1d(1, 32, 15, stride=1, padding=7)))
        self.convs.append(wn(nn.Conv1d(32, 128, 41, stride=2, groups=4, padding=20)))
        self.convs.append(wn(nn.Conv1d(128, 256, 41, stride=2, groups=4, padding=20)))
        self.convs.append(wn(nn.Conv1d(256, 512, 41, stride=2, groups=16, padding=20)))
        self.convs.append(wn(nn.Conv1d(512, 1024, 41, stride=4, groups=32, padding=20)))
        self.convs.append(wn(nn.Conv1d(1024, 1024, 41, stride=1, groups=32, padding=20)))
        self.convs.append(wn(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)))
        self.convs.append(wn(nn.Conv1d(1024, 1, 3, stride=1, padding=1)))
        for conv in self.convs:
            init_weights(conv)
    
    def forward(self, x):
        features = []
        if self.pooling != None:
            x = self.pooling(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            if i != len(self.convs) - 1:
                x = lrlu(x)
            features.append(x)
        x = torch.flatten(x, 1)
        return x, features

class MSD(nn.Module):
    def __init__(self, poolings):
        super(MSD, self).__init__()
        self.poolings = poolings
        self.sds = nn.ModuleList()
        for pooling in self.poolings:
            self.sds.append(SDiscriminator(pooling))
        
    def forward(self, real, gen):
        real_features = []
        gen_features = []
        real_preds = []
        gen_preds = []
        for i in range(len(self.poolings)):
            real_pred, real_feature = self.sds[i](real)
            gen_pred, gen_feature = self.sds[i](gen)
            real_preds.append(real_pred)
            gen_preds.append(gen_pred)
            real_features.append(real_feature)
            gen_features.append(gen_feature)
        return real_preds, gen_preds, real_features, gen_features

                          
def gen_feature_loss (real_features, gen_features):
    loss = 0
    layers = len(real_features) * len(real_features[0])
    for i in range(len(real_features)):
        for j in range(len(real_features[i])):
            loss += torch.mean(torch.abs(real_features[i][j] - gen_features[i][j]))
    return loss / layers

def disc_adv_loss (real_preds, gen_preds):
    loss = 0
    real_losses = []
    gen_losses = []
    for real_pred, gen_pred in zip(real_preds, gen_preds):
        real_loss = torch.mean((real_pred-1)**2)
        gen_loss = torch.mean(gen_pred**2)
        real_losses.append(real_loss)
        gen_losses.append(gen_loss)
        loss += real_loss + gen_loss
    real_loss = torch.tensor(real_losses)
    gen_loss = torch.tensor(gen_losses)
    loss /= 2 * len(real_preds)
    return loss, real_losses, gen_losses

def gen_adv_loss (gen_preds):
    loss = 0
    gen_losses = []
    for gen_pred in gen_preds:
        gen_loss = torch.mean((gen_pred-1)**2)
        gen_losses.append(gen_loss)
        loss += gen_loss
    loss /= len(gen_preds)
    return loss

def gen_spec_loss (real, gen, cfg):
    real = real.to('cpu')
    gen = gen.to('cpu')
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=cfg['sample_rate'], n_fft=cfg['n_fft'], win_length=cfg['win_length'], hop_length=cfg['hop_length'], n_mels=cfg['n_mels'], center=cfg['center'])
    real_mel = mel(real)
    gen_mel = mel(gen)
    loss = F.l1_loss(real_mel, gen_mel)
    real = real.to(cfg['device'])
    gen = gen.to(cfg['device'])
    return loss
                
                
        
        


