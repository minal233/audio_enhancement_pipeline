# 🎵 AI-Powered Music Emotion Enhancer

An intelligent audio enhancement tool that:
-Analyzes the emotional content of music (valence & arousal)
-Applies tailored DSP effects to make the song sound better while matching its mood
-Preserves full **stereo** and **vocal clarity**
-Features a beautiful **Streamlit web interface**

Built with PyTorch, torchaudio, SciPy, and trained on the PMEmo dataset.

## Features

- Emotion prediction (Happy/Energetic → bright & punchy, Sad/Calm → warm & intimate)
- Musical EQ: vocal-focused midrange boost + mood-based high/low shelf
- Soft saturation and headroom for natural sound
- Full stereo processing
- Web app: upload any song, compare original vs enhanced, download result

## Demo

Run the web app locally:

```bash
streamlit run app.py
```

## Setup & Installation

### 1. Clone the repo

```Bash
git clone https://github.com/yourusername/audio_enhancement_pipeline.git
cd audio_enhancement_pipeline
```

### 2. Create virtual environment

```Bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```Bash
pip install torch torchaudio numpy scipy pandas streamlit scikit-learn matplotlib tqdm
```

### 4. Run the app

```Bash
streamlit run app.py
```

## Credits

Emotion model trained on the [PMEmo](https://github.com/HuiZhangDB/PMEmo)
Built step-by-step with lots of debugging and polishing 🔥

Enjoy making your music sound emotionally smarter! 🎧 ✨