from imports import *
import torchaudio
import matplotlib.pyplot as plt

import torch
import torchaudio

import torchaudio
import torch

def compute_dc_component(batch, window_size, hop_length):
    # Ensure batch is at least 3D (batch_size, num_channels, num_samples)
    if batch.dim() == 2:
        batch = batch.unsqueeze(1)

    # Unfold the signal along the last dimension
    unfolded = batch.unfold(-1, window_size, hop_length)
    
    # Compute the mean (DC component) of each window
    dc_component = unfolded.mean(dim=-1)

    # Convert to DB scale
    dc_component_db = torchaudio.transforms.AmplitudeToDB()(dc_component)

    dc_component_db = dc_component_db.unsqueeze(2)
    
    return dc_component_db

def pad_dc_component(dc_component, target_length):
    batch_size, num_channels, _, num_frames = dc_component.shape
    if num_frames < target_length:
        padding_length = target_length - num_frames
        padding = dc_component[:, :, :, :padding_length]
        dc_component = torch.cat((dc_component, padding), dim=-1)
    return dc_component

def convert_batch_to_log_mel_stft(batch, sample_rate, n_fft=2048, window_size=1024, n_mels=128):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_mels*16, win_length=window_size, hop_length=window_size // 4, n_mels=n_mels)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_transform(batch))
    
    # Compute DC component
    dc_component = compute_dc_component(batch, window_size, window_size // 4)

    # Pad dc_component if necessary
    target_length = log_mel_spectrogram.size(-1)
    dc_component = pad_dc_component(dc_component, target_length)
    
    # Concatenate dc_component as the first row of log_mel_spectrogram
    for i in range(0):
        log_mel_spectrogram = torch.cat((dc_component, log_mel_spectrogram), dim=2)
    
    return log_mel_spectrogram

def band_pass(path, low_freq, high_freq):
    # Create a bandpass filter

    # Load your signal
    signal, sr = librosa.load(path, sr=44100)
    print(sr)

    # Compute the STFT
    D = librosa.stft(signal)

    low_taper_width = 40  # Width of the tapering in Hz
    high_taper_width = 12000  # Width of the tapering in Hz

    # Create the frequency mask with smooth transitions
    fft_frequencies = librosa.fft_frequencies(sr=44100)
    mask = np.ones_like(fft_frequencies)  # Initialize the mask with ones

    # Low-frequency taper
    low_taper =  (1 + np.cos(np.pi * (fft_frequencies - low_freq) / low_taper_width)) / 2
    mask *= np.where((fft_frequencies >= (low_freq - low_taper_width)) & (fft_frequencies < low_freq), low_taper, 1)

    # Passband
    mask *=  np.where((fft_frequencies >= low_freq) & (fft_frequencies <= high_freq), 1, 1)

    # High-frequency taper
    high_taper = 0.3 + 0.7 * (1 + np.cos(np.pi * (fft_frequencies - high_freq) / high_taper_width)) / 2
    mask *= np.where((fft_frequencies > high_freq) & (fft_frequencies <= (high_freq + high_taper_width)), high_taper, 1)

    # Set mask to 0.5 outside the taper ranges
    mask = np.where(fft_frequencies > (high_freq + high_taper_width), 0.3, mask)
    mask = np.where(fft_frequencies < (low_freq - low_taper_width), 0.5, mask)

    # Ensure the values are in the correct range
    mask = np.clip(mask, 0, 1)

    plt.figure()
    plt.plot(fft_frequencies, mask)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Mask')
    plt.grid(True)
    plt.show()



    # Apply the frequency mask
    D_band_pass = D * mask[:, np.newaxis]

    # Inverse STFT
    filtered_signal = librosa.istft(D_band_pass)


    # Save or play the filtered signal
    sf.write('../filtered_signal.wav', filtered_signal, 44100)

    return filtered_signal


#band_pass('../model_results/validation_predictions_epoch_43.wav', 40, 5000)