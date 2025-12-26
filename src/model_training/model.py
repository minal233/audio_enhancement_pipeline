import torch.nn as nn


class EmotionRegressor(nn.Module):
    def __init__(self, input_dim=32, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)  # Valence, Arousal
        )

    def forward(self, x):
        return self.network(x)
