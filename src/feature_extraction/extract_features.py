import os
import torch
import torchaudio
import numpy as np

# Settings (same as training)
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
WIN_LENGTH = 1024


def extract_features_safe(waveform):
    """Extract features with NaN safety"""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    features = {}

    # MFCC
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={'n_fft': N_FFT, 'hop_length': HOP_LENGTH,
                   'win_length': WIN_LENGTH}
    )
    mfcc = mfcc_transform(waveform).squeeze(0)
    features['mfcc_mean'] = mfcc.mean(dim=1).cpu().numpy()
    features['mfcc_std'] = np.nan_to_num(
        mfcc.std(dim=1).cpu().numpy(), nan=0.0)

    # Centroid
    centroid_transform = torchaudio.transforms.SpectralCentroid(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
    )
    centroid = centroid_transform(waveform).squeeze(0)
    features['centroid_mean'] = np.array(
        [centroid.mean().item() if centroid.numel() > 0 else 0])
    features['centroid_std'] = np.array(
        [centroid.std().item() if centroid.numel() > 1 else 0])

    # Rolloff (safe)
    stft = torch.stft(
        waveform.squeeze(0), n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH), center=True, onesided=True, return_complex=True
    )
    magnitude = stft.abs()**2
    total_energy = magnitude.sum(dim=0, keepdim=True)
    cumulative = torch.cumsum(magnitude, dim=0)
    target = 0.85 * total_energy
    rolloff_idx = torch.sum((cumulative < target), dim=0).long()
    rolloff_idx = torch.clamp(rolloff_idx, max=magnitude.shape[0]-1)
    freq_bins = torch.linspace(0, SAMPLE_RATE/2, steps=magnitude.shape[0])
    rolloff = freq_bins[rolloff_idx]
    features['rolloff_mean'] = np.array(
        [rolloff.mean().item() if rolloff.numel() > 0 else 4000])
    features['rolloff_std'] = np.array(
        [rolloff.std().item() if rolloff.numel() > 1 else 0])

    # ZCR & RMS (global safe)
    zcr = ((waveform[:, :-1] * waveform[:, 1:]) < 0).float().mean()
    features['zcr_mean'] = np.array([zcr.item()])

    rms = torch.sqrt(torch.mean(waveform**2))
    features['rms_mean'] = np.array([rms.item()])

    vec = np.concatenate([v for v in features.values()])
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec
