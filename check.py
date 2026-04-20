from torchvision import datasets
ds = datasets.ImageFolder('data/PlantVillage')
print(ds.classes)
print(len(ds.classes))