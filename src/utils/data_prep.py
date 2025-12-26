import os
from pydub import AudioSegment
import pandas as pd

# Target format: 16kHz mono WAV, normalized volume
TARGET_SR = 16000
TARGET_CHANNELS = 1  # Mono


def convert_to_wav(input_path, output_path):
    """Convert single MP3 to normalized mono 16kHz WAV"""
    audio = AudioSegment.from_mp3(input_path)  # pydub handles MP3 via ffmpeg
    audio = audio.set_frame_rate(TARGET_SR).set_channels(TARGET_CHANNELS)
    audio = audio.normalize()  # Peak normalize to -1 dBFS for consistency
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio.export(output_path, format="wav")
    print(
        f"Converted: {os.path.basename(input_path)} → {os.path.basename(output_path)}")


def batch_preprocess(raw_root_dir, processed_dir):
    """Convert all chorus MP3s to WAVs, preserving song ID in filename"""
    chorus_dir = os.path.join(raw_root_dir, 'chorus')
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(chorus_dir):
        raise FileNotFoundError(f"Chorus folder not found at {chorus_dir}")

    mp3_files = [f for f in os.listdir(
        chorus_dir) if f.lower().endswith('.mp3')]
    print(f"Found {len(mp3_files)} MP3 files in chorus folder.")

    for mp3_file in sorted(mp3_files):
        input_path = os.path.join(chorus_dir, mp3_file)
        # Preserve original filename but change extension (e.g., 1.mp3 → 1.wav)
        wav_file = os.path.splitext(mp3_file)[0] + '.wav'
        output_path = os.path.join(processed_dir, wav_file)
        convert_to_wav(input_path, output_path)

    print("All conversions complete!")


def load_static_annotations(raw_root_dir):
    """Load and display static annotations with correct column names"""
    annotations_path = os.path.join(
        raw_root_dir, 'annotations', 'static_annotations.csv')
    if not os.path.exists(annotations_path):
        print("static_annotations.csv not found!")
        return None

    df = pd.read_csv(annotations_path)
    print("\nStatic Annotations Sample (first 5 rows):")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")

    # Correct column names based on actual file
    arousal_col = 'Arousal(mean)'
    valence_col = 'Valence(mean)'

    print(
        f"Arousal range: {df[arousal_col].min():.4f} to {df[arousal_col].max():.4f}")
    print(
        f"Valence range: {df[valence_col].min():.4f} to {df[valence_col].max():.4f}")
    print(f"Number of songs with annotations: {len(df)}")

    return df


if __name__ == "__main__":
    raw_dir = 'data/raw/pmemo'          # Adjust if needed
    processed_dir = 'data/processed/pmemo'

    batch_preprocess(raw_dir, processed_dir)
    load_static_annotations(raw_dir)    # Now no error
