import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────
DATA_DIR    = "data/PlantVillage"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 0.001
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Data loading ─────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
NUM_CLASSES  = len(full_dataset.classes)
print(f"Classes: {NUM_CLASSES}")
print(f"Total images: {len(full_dataset)}")

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {train_size} | Val: {val_size}")

# ── MobileNetV2 Transfer Learning ────────────────────
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze all base layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, NUM_CLASSES)
)

model = model.to(DEVICE)
print("MobileNetV2 loaded with frozen base layers")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Phase 1: Train head only ───────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

print("\n--- Phase 1: Training classifier head ---")
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    train_accs.append(correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / val_total)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} | "
          f"Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

# ── Phase 2: Unfreeze top layers and fine-tune ─────────
print("\n--- Phase 2: Fine-tuning top layers ---")
for param in model.features[-3:].parameters():
    param.requires_grad = True

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001
)

for epoch in range(3):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accs.append(correct / total)
    val_accs.append(val_correct / val_total)

    print(f"Fine-tune Epoch {epoch+1}/3 | "
          f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

# ── Save model ────────────────────────────────────────
torch.save(model.state_dict(), "models/mobilenet_transfer.pth")
print("\nModel saved to models/mobilenet_transfer.pth")

# ── Plot results ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train", color="#7F77DD")
ax1.plot(val_losses,   label="Val",   color="#D85A30")
ax1.axvline(x=EPOCHS-1, color="gray", linestyle="--", linewidth=0.8, label="Fine-tune starts")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

ax2.plot(train_accs, label="Train", color="#7F77DD")
ax2.plot(val_accs,   label="Val",   color="#D85A30")
ax2.axvline(x=EPOCHS-1, color="gray", linestyle="--", linewidth=0.8, label="Fine-tune starts")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

plt.suptitle("MobileNetV2 Transfer Learning — Training Results", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/transfer_results.png", dpi=150)
plt.show()
print("Saved: outputs/transfer_results.png")