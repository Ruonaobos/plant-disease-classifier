import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Point to dataset
DATA_DIR = "data/PlantVillage"

# Get all class names - filter out any nested folders that aren't image classes
classes = sorted([
    c for c in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, c)) and c != "PlantVillage"
])

print(f"Total classes: {len(classes)}")
print(classes)

# Count images per class
counts = {}
for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    counts[cls] = len(os.listdir(cls_path))

total = sum(counts.values())
print(f"\nTotal images: {total}")

# Plot class distribution
plt.figure(figsize=(14, 6))
plt.bar(range(len(counts)), list(counts.values()), color="#378ADD")
plt.xticks(range(len(counts)), list(counts.keys()), rotation=90, fontsize=7)
plt.title("Images per class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.show()
print("Saved: outputs/class_distribution.png")

# Show sample images from 6 classes
sample_classes = classes[:6]
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes = axes.flatten()

for i, cls in enumerate(sample_classes):
    cls_path = os.path.join(DATA_DIR, cls)
    img_file = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))][0]
    img = mpimg.imread(os.path.join(cls_path, img_file))
    axes[i].imshow(img)
    axes[i].set_title(cls.replace("_", " "), fontsize=8)
    axes[i].axis("off")

plt.suptitle("Sample images from dataset", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/sample_images.png", dpi=150)
plt.show()
print("Saved: outputs/sample_images.png")