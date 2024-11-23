from imports import *
from models import Generator

def infer_example(gen, lq_batch, device):
    lq_batch = lq_batch.to(device)
    gen.eval()
    output = gen(lq_batch)
    return output


def infer(cfg, lq_folder):
    device = torch.device(cfg['device'])
    gen = Generator(cfg).to(device)
    gen.load_state_dict(torch.load('Models/HiFi-GAN/Best/gen.pt'))
    gen.eval()
    outputs = []
    lq_files = os.listdir(lq_folder)
    i = 0
    with torch.no_grad():
        for lq_file in lq_files:
            if not lq_file.endswith('.pt'):
                continue
            lq = torch.load(os.path.join(lq_folder, lq_file))
            lq = lq.to(device)
            output = infer_example(gen, lq, device)
            outputs.append(output)
            i += 1
            if i == 100:
                break
    return outputs

def concat_outputs(outputs, cfg):
    output = torch.cat(outputs, dim=2)
    return output
cfg = {
        'segment_size': 16384,
        'sample_rate': 44100,
        'n_fft': 4096,
        'win_length': 1024,
        'hop_length': 520,
        'n_mels': 256,
        'center': True,
        'device': 'cpu',
        'train_lq_path': 'Data/LQ/24/pilot/Chunks',
        'train_hq_path': 'Data/HQ/pilot/Chunks',
        'valid_lq_path': 'Data/LQ/24/pilot/Chunks',
        'valid_hq_path': 'Data/HQ/pilot/Chunks',
        'epochs': 100,
        'lr': 0.0001,
        'beta1': 0.8,
        'beta2': 0.99,
        'load': None,
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

outputs = infer(cfg, 'Data/LQ/24/train/Chunks')
chunk_0 = outputs[0][0]
chunk_0_hq, _ = torchaudio.load('Data/HQ Light/train/Chunks/0.wav')
chunk_0_hq = torch.mean(chunk_0_hq, dim=0, keepdim=True)
print(chunk_0.shape)
print(chunk_0_hq.shape)

plt.figure(figsize=(20, 4))
plt.plot(chunk_0[0].cpu().numpy())
plt.plot(chunk_0_hq[0].numpy())
plt.legend(['Generated', 'Ground Truth'])
plt.show()

outputs = concat_outputs(outputs, cfg)
print(outputs.shape)
outputs = outputs.squeeze(0)
print(outputs.shape)
torchaudio.save('Data/Generated.wav', outputs, cfg['sample_rate'])

outputs = infer(cfg, 'Data/LQ/24/test/Chunks')
chunk_0 = outputs[0][0]
chunk_0_hq, _ = torchaudio.load('Data/HQ Light/test/Chunks/0.wav')
chunk_0_hq = torch.mean(chunk_0_hq, dim=0, keepdim=True)
print(chunk_0.shape)
print(chunk_0_hq.shape)

plt.figure(figsize=(20, 4))
plt.plot(chunk_0[0].cpu().numpy())
plt.plot(chunk_0_hq[0].numpy())
plt.legend(['Generated', 'Ground Truth'])
plt.show()