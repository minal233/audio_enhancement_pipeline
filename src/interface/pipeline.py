import os
import torch
import torchaudio
from scipy.signal import butter, sosfilt
import numpy as np
from src.model_training.model import EmotionRegressor
from src.feature_extraction.extract_features import extract_features_safe

# Model setup
MODEL_PATH = 'models/checkpoints/best_emotion_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EmotionRegressor(input_dim=32).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def predict_emotion(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    # Use mono for feature extraction (emotion is content-based)
    mono = waveform.mean(dim=0, keepdim=True)
    features = extract_features_safe(mono).astype(np.float32)
    features_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(features_tensor).cpu().numpy()[0]

    valence, arousal = pred
    return {'valence': float(valence), 'arousal': float(arousal)}


def apply_stereo_dsp_effects(waveform, valence, arousal):
    fs = 16000
    nyq = fs / 2

    # Split channels
    samples_l = waveform[0].numpy()
    samples_r = waveform[1].numpy(
    ) if waveform.shape[0] > 1 else samples_l.copy()

    # Normalize per channel
    for ch in [samples_l, samples_r]:
        max_abs = np.max(np.abs(ch))
        if max_abs > 0:
            ch /= max_abs

    def high_shelf(freq, gain_db, order=2):
        freq = min(freq, nyq * 0.95)
        sos = butter(order, freq / nyq, btype='high', output='sos')
        return sos, 10 ** (gain_db / 20)

    def low_shelf(freq, gain_db, order=2):
        sos = butter(order, freq / nyq, btype='low', output='sos')
        return sos, 10 ** (gain_db / 20)

    def peaking(freq, gain_db, Q=1.4):
        sos = butter(2, [freq / Q / nyq, freq * Q / nyq],
                     btype='band', output='sos')
        return sos, 10 ** (gain_db / 20)

    # Always: Gentle vocal clarity boost
    mid_sos, mid_gain = peaking(3000, +3.0, Q=1.2)
    samples_l = sosfilt(mid_sos, samples_l) * mid_gain
    samples_r = sosfilt(mid_sos, samples_r) * mid_gain

    # Emotion-based enhancement
    if arousal > 0.65 and valence > 0.6:
        sos, gain = high_shelf(5000, +4.0)
        effect = "Joyful & Energetic: Sparkle + clarity"
    elif arousal > 0.6 and valence < 0.4:
        sos, gain = high_shelf(3500, +5.0)
        effect = "Intense: Power & presence"
    elif arousal < 0.4 and valence > 0.6:
        sos, gain = low_shelf(200, +4.5)
        samples_l = sosfilt(sos, samples_l) * gain
        samples_r = sosfilt(sos, samples_r) * gain
        sos2, gain2 = high_shelf(7000, +2.0)
        effect = "Calm Joy: Warmth + soft air"
    elif arousal < 0.4 and valence < 0.4:
        sos, gain = low_shelf(250, +5.0)
        samples_l = sosfilt(sos, samples_l) * gain
        samples_r = sosfilt(sos, samples_r) * gain
        sos2, _ = high_shelf(6000, -7.0)
        samples_l = sosfilt(sos2, samples_l)
        samples_r = sosfilt(sos2, samples_r)
        effect = "Melancholic: Intimate warmth"
    else:
        sos, gain = high_shelf(6000, +2.0)
        effect = "Balanced polish with vocal focus"

    # Apply main effect if defined
    if 'sos' in locals():
        samples_l = sosfilt(sos, samples_l) * gain
        samples_r = sosfilt(sos, samples_r) * gain

    # Recombine
    enhanced = np.stack([samples_l, samples_r])

    # Soft saturation + headroom
    enhanced = np.tanh(enhanced * 1.05) / 1.05
    peak = np.max(np.abs(enhanced))
    if peak > 0:
        enhanced /= peak
        enhanced *= 0.97

    enhanced_tensor = torch.from_numpy(enhanced).float()
    return enhanced_tensor, effect


def enhance_audio_stereo(input_path, output_path=None):
    # Always save as .wav for reliability
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_enhanced.wav"

    print(f"Processing: {input_path}")

    try:
        emotions = predict_emotion(input_path)
        print(
            f"Predicted → Valence: {emotions['valence']:.3f} | Arousal: {emotions['arousal']:.3f}")

        waveform, sr = torchaudio.load(input_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        enhanced_waveform, effect_desc = apply_stereo_dsp_effects(
            waveform, emotions['valence'], emotions['arousal'])
        print(f"→ {effect_desc}")

        torchaudio.save(output_path, enhanced_waveform, 16000, format="wav")
        print(f"Enhanced saved: {output_path}\n")

        return output_path, emotions, effect_desc

    except Exception as e:
        print(f"Error during processing: {e}")
        raise e  # Let Streamlit handle display
