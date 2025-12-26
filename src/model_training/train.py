import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from model import EmotionRegressor

# Paths
FEATURES_DIR = 'data/processed/features/pmemo'
X_NPY = os.path.join(FEATURES_DIR, 'X_features.npy')
IDS_NPY = os.path.join(FEATURES_DIR, 'song_ids.npy')
DATASET_CSV = os.path.join(FEATURES_DIR, 'dataset_with_features.csv')
CHECKPOINT_DIR = 'models/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load features and IDs
X_full = np.load(X_NPY).astype(np.float32)
song_ids_full = np.load(IDS_NPY)

print(f"Loaded {len(X_full)} feature vectors (dim: {X_full.shape[1]})")

# Load merged dataset
df = pd.read_csv(DATASET_CSV)
df['musicId'] = df['musicId'].astype(int)
print(f"Loaded {len(df)} songs with annotations")

# Map song_id to feature index
id_to_idx = {song_id: idx for idx, song_id in enumerate(song_ids_full)}

# Get valid indices
valid_indices = []
for sid in df['musicId']:
    if sid in id_to_idx:
        valid_indices.append(id_to_idx[sid])
    else:
        print(f"Warning: No features for song ID {sid}")

X = X_full[valid_indices]

# Critical: Handle NaN in features
nan_mask = np.isnan(X).any(axis=1)
num_nan = np.sum(nan_mask)
if num_nan > 0:
    print(f"Found {num_nan} songs with NaN features. Removing them...")
    X = X[~nan_mask]
    # Rough align; better to use indices
    df = df[~df['musicId'].isin(df['musicId'][nan_mask])]
    # Re-align df with cleaned X
    kept_ids = [song_ids_full[i] for i in valid_indices]
    kept_ids = np.array(kept_ids)[~nan_mask]
    df = pd.DataFrame({'musicId': kept_ids})
    df = df.merge(pd.read_csv(DATASET_CSV), on='musicId')  # Re-merge clean

labels = df[['Arousal(mean)', 'Valence(mean)']].values.astype(np.float32)

print(f"Final clean dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Labels shape: {labels.shape}")

if X.shape[0] < 100:
    raise ValueError("Too few valid samples after cleaning!")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, labels, test_size=0.2, random_state=42
)


class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = AudioDataset(X_train, y_train)
val_dataset = AudioDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = EmotionRegressor(input_dim=X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training
num_epochs = 200
best_val_loss = float('inf')
patience = 20
wait = 0

train_losses = []
val_losses = []

print("\nStarting training...\n")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_x.size(0)

    avg_train_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    val_rmse = np.sqrt(mean_squared_error(trues, preds))

    print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val RMSE: {val_rmse:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), os.path.join(
            CHECKPOINT_DIR, 'best_emotion_model.pth'))
        print("   >>> Best model saved!")
    else:
        wait += 1
        if wait >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Emotion Model Training History')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_curve.png'))
plt.show()

print(f"\nTraining complete!")
print(f"Best model: {os.path.join(CHECKPOINT_DIR, 'best_emotion_model.pth')}")
print(f"Final Val RMSE: {val_rmse:.4f} (target < 0.20)")
