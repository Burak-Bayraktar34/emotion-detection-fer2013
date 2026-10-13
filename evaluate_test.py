import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import EmotionCNN

from sklearn.metrics import classification_report, confusion_matrix

test_yolu = "dataset/test" 

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(), 
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224))
])

test_dataset = ImageFolder(root=test_yolu, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(len(test_dataset))
print(test_dataset.classes)

gercek_model = EmotionCNN()
gercek_model.load_state_dict(torch.load("model.pth"))
gercek_model.eval()

tum_gercekler = []
tum_tahminler = []
with torch.no_grad():
    for girdi, etiket in test_loader:
        cikti = gercek_model(girdi)
        _, tahmin = torch.max(cikti, 1)
        tum_gercekler.extend(etiket.numpy())
        tum_tahminler.extend(tahmin.numpy())

print(classification_report(tum_gercekler, tum_tahminler, target_names=test_dataset.classes))
print(confusion_matrix(tum_gercekler, tum_tahminler))

