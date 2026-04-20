# Plant Disease Classifier
### Deep Learning & Transfer Learning | PyTorch

A machine learning project that classifies plant diseases from leaf images using two approaches:
1. **Custom CNN** — built and trained from scratch
2. **Transfer Learning** — MobileNetV2 pre-trained on ImageNet, fine-tuned on plant data

---

## Dataset
- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes:** 15 (covering Pepper, Potato, and Tomato diseases)
- **Total Images:** 20,639
- **Split:** 80% train / 20% validation

---

## Techniques Demonstrated
| Technique | Description |
|---|---|
| Deep Learning | Custom 3-layer CNN built with PyTorch |
| Transfer Learning | MobileNetV2 fine-tuned on domain-specific data |
| Data Preprocessing | Normalization, resizing, train/val split |
| Model Evaluation | Loss curves, accuracy curves, model comparison |

---

## Results
| Model | Val Accuracy |
|---|---|
| CNN from Scratch | 95.5% |
| MobileNetV2 (Transfer Learning) | 98.3% |

---

## Project Structure
plant-disease-classifier/
├── data/               # Dataset (not tracked in git)
├── notebooks/
│   ├── 01_eda.py       # Exploratory data analysis
│   ├── 02_cnn.py       # CNN from scratch
│   └── 03_transfer.py  # Transfer learning
├── models/             # Saved model weights
├── outputs/            # Charts and visualizations
├── requirements.txt
└── README.md

--- 

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/ruonaobos/plant-disease-classifier.git
cd plant-disease-classifier

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
kaggle datasets download -d emmarex/plantdisease -p data/ --unzip

# 5. Run scripts in order
python notebooks/01_eda.py
python notebooks/02_cnn.py
python notebooks/03_transfer.py
```

---

## Tools & Libraries
- Python 3.11
- PyTorch & TorchVision
- Matplotlib
- Scikit-learn
- NumPy

---

