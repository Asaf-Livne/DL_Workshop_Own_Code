from imports import *
from dataloader import PairedWavDataset as DS
from wavenet_train import ESR_loss as ESR
'''

dataset = DS('Data/LQ/3/train', 'Data/HQ/train', 44100)
print(dataset.__len__())
print(np.array(dataset.hq_chunks).shape)
dl = DataLoader(dataset, batch_size=10, shuffle=True)
print('fini')

flag = 0

for lq, hq in dl:
    if flag == 0:
        # save the first batch
        torchaudio.save('lq.wav', lq[0], 44100)
        torchaudio.save('hq.wav', hq[0], 44100)
        flag = 1
'''
# compare the two files
lq, _ = torchaudio.load('Data/LQ/3/train/Music Delta - Hendrix.wav')
hq, _ = torchaudio.load('Data/HQ/train/Music Delta - Hendrix.wav')
lq = lq[0] + lq[1]
hq = hq[0] + hq[1]

lq = np.array(lq)
hq = np.array(hq)

plt.figure()
plt.plot(lq)
plt.plot(hq)
plt.show()
# Create STFTs 
lq = librosa.stft(lq, n_fft=1024, hop_length=512)
hq = librosa.stft(hq, n_fft=1024, hop_length=512)
# Log magnitude
lq = np.log1p(np.abs(lq))
hq = np.log1p(np.abs(hq))
print(lq.shape) 

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(lq, aspect='auto', origin='lower')
plt.colorbar()
plt.title('LQ')
plt.subplot(1, 2, 2)
plt.imshow(hq, aspect='auto', origin='lower')
plt.colorbar()
plt.title('HQ')
plt.show()


