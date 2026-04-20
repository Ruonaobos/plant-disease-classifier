import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Reproducibility ───────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────
DATA_DIR   = "data/PlantVillage"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 0.001
IMG_SIZE   = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Data pipeline with augmentation ──────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

full_dataset = datasets.ImageFolder(DATA_DIR)
classes      = full_dataset.classes
NUM_CLASSES  = len(classes)
labels       = [s[1] for s in full_dataset.samples]
print(f"Classes: {NUM_CLASSES} | Total images: {len(full_dataset)}")

# ── Stratified split ──────────────────────────────────
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

train_ds = Subset(full_dataset, train_idx)
val_ds   = Subset(full_dataset, val_idx)

train_ds.dataset.transform = train_transform
val_ds.dataset.transform   = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── CNN Model ─────────────────────────────────────────
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model     = PlantCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# ── Training with early stopping ──────────────────────
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_val_loss  = float('inf')
patience       = 3
patience_count = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels_batch in train_loader:
        images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct      += (outputs.argmax(1) == labels_batch).sum().item()
        total        += labels_batch.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            outputs   = model(images)
            loss      = criterion(outputs, labels_batch)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels_batch).sum().item()
            val_total   += labels_batch.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / val_total)
    scheduler.step(val_losses[-1])

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Early stopping
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), "models/cnn_scratch_best.pth")
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print("Best model saved to models/cnn_scratch_best.pth")

# ── Evaluation ────────────────────────────────────────
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix — CNN from Scratch")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/cnn_confusion_matrix.png", dpi=150)
plt.show()
print("Saved: outputs/cnn_confusion_matrix.png")

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train", color="#378ADD")
ax1.plot(val_losses,   label="Val",   color="#D85A30")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(train_accs, label="Train", color="#378ADD")
ax2.plot(val_accs,   label="Val",   color="#D85A30")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.suptitle("CNN from Scratch — Training Results", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/cnn_results.png", dpi=150)
plt.show()
print("Saved: outputs/cnn_results.png")