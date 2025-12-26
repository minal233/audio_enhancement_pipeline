import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths – adjust only if your folder structure differs
RAW_DIR = 'data/raw/pmemo'
ANNOTATIONS_PATH = os.path.join(
    RAW_DIR, 'annotations', 'static_annotations.csv')


def load_annotations():
    """Load static valence/arousal annotations"""
    if not os.path.exists(ANNOTATIONS_PATH):
        raise FileNotFoundError(f"Annotations not found at {ANNOTATIONS_PATH}")

    df = pd.read_csv(ANNOTATIONS_PATH)
    print(f"Loaded annotations for {len(df)} songs")
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    return df


def plot_va_distribution(df):
    """Plot Valence vs Arousal scatter"""
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Arousal(mean)'], df['Valence(mean)'],
                alpha=0.7, c='purple', edgecolors='white', s=80)
    plt.xlabel('Arousal (mean)', fontsize=14)
    plt.ylabel('Valence (mean)', fontsize=14)
    plt.title(
        'PMEmo Dataset: Valence-Arousal Distribution (794 Music Clips)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add quadrant labels
    plt.text(0.75, 0.75, 'High Energy / Positive',
             fontsize=12, ha='center', color='green')
    plt.text(0.25, 0.75, 'High Energy / Negative',
             fontsize=12, ha='center', color='red')
    plt.text(0.25, 0.25, 'Low Energy / Negative',
             fontsize=12, ha='center', color='blue')
    plt.text(0.75, 0.25, 'Low Energy / Positive',
             fontsize=12, ha='center', color='gray')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    annotations_df = load_annotations()
    plot_va_distribution(annotations_df)

    # Summary statistics
    print("\nSummary Statistics:")
    print(annotations_df[['Arousal(mean)', 'Valence(mean)']].describe())
