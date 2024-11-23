from imports import *
from dataloader import PairedWavDataset as DS
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

def ESR(x, y):
    y2 = y**2
    xmy2 = (x - y)**2
    return np.sqrt(np.sum(xmy2) / np.sum(y2))
# compare the two files
lq, _ = torchaudio.load('Results/WaveNet/validation/E1_LQ.wav')
hq, _ = torchaudio.load('Results/WaveNet/validation/E1_HQ.wav')
gen, _ = torchaudio.load('Results/WaveNet/validation/E1_GEN.wav')


lq = np.array(lq)
hq = np.array(hq)
gen = np.array(gen)

#print(ESR(lq[:-1], hq))


lq = lq[0]
hq = hq[0]
gen = gen[0]

lq = lq[-len(gen):] 
hq = hq[-len(gen):]

print(len(lq), len(hq), len(gen))

plt.figure()
plt.plot(lq)
plt.plot(hq)
plt.plot(gen)
plt.show()
# Create STFTs 
lq = librosa.stft(lq, n_fft=1024*16, hop_length=512, win_length=1024)
hq = librosa.stft(hq, n_fft=1024*16, hop_length=512, win_length=1024)
gen = librosa.stft(gen, n_fft=1024*16, hop_length=512, win_length=1024)
# Log magnitude
lq = np.log1p(np.abs(lq))
hq = np.log1p(np.abs(hq))
gen = np.log1p(np.abs(gen))
print(ESR(lq, hq))
print(ESR(lq, gen))
print(ESR(hq, gen))

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(lq, aspect='auto', origin='lower')
plt.colorbar()
plt.title('LQ')
plt.subplot(1, 3, 2)
plt.imshow(hq, aspect='auto', origin='lower')
plt.colorbar()
plt.title('HQ')
plt.subplot(1, 3, 3)
plt.imshow(gen, aspect='auto', origin='lower')
plt.colorbar()
plt.title('GEN')
plt.show()


