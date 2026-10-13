import torch
import torchvision
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Subset
from model_efficientnet import create_efficientnet_model
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from focal_loss import FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Modeli oluştur (henüz eğitilmemiş, ağırlıklar rastgele)
model = create_efficientnet_model(num_classes=7)
model = model.to(device)
model.eval()

train_yolu = "dataset/train"

# İki ayrı transform: train için augmentation'lı, validation için sade
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet expects 224x224 input
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet expects 224x224 input
])

# Aynı klasörden, iki farklı transform ile iki dataset oluştur
train_dataset_aug = ImageFolder(root=train_yolu, transform=train_transform)
train_dataset_plain = ImageFolder(root=train_yolu, transform=val_transform)

print(len(train_dataset_aug))
print(train_dataset_aug.classes)

# Rastgeleliği sabitle
torch.manual_seed(42)

# Sadece indeksleri böl (görüntülere henüz dokunmadan)
train_indices, val_indices = random_split(range(len(train_dataset_aug)), [0.85, 0.15])

# Subset ile: train -> augmentation'lı dataset, val -> sade dataset, aynı indekslerle
train_set = Subset(train_dataset_aug, train_indices.indices)
val_set = Subset(train_dataset_plain, val_indices.indices)

print(len(train_set))
print(len(val_set))

# DataLoader
trainset = DataLoader(train_set, batch_size=16, shuffle=True)
valset = DataLoader(val_set, batch_size=16, shuffle=False)

# Class weighting (dengesizlik için)
sinif_sayilari = [3995, 436, 4097, 7215, 4965, 4830, 3171]
toplam = sum(sinif_sayilari)
agirliklar = []
for sayi in sinif_sayilari:
    agirlik = toplam / (len(sinif_sayilari) * sayi)
    agirliklar.append(agirlik)
senin_agirlik_tensorun = torch.tensor(agirliklar, dtype=torch.float32)
senin_agirlik_tensorun = senin_agirlik_tensorun.to(device)
print(agirliklar)

criterion = FocalLoss(gamma=2.0, weight=None)
optimizer = optim.Adam([
    {'params': model.features[-3].parameters(), 'lr': 0.0001},
    {'params': model.classifier.parameters(), 'lr': 0.001}
])

train_losses = []
val_losses = []

en_iyi_val_loss = float('inf')

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# Training Loop
for epoch in range(20):
    toplam_loss = 0
    for images, labels in trainset:
         images, labels = images.to(device), labels.to(device)   # 1. ÖNCE GPU'ya taşı
         images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)   # 2. SONRA mixup uygula
         optimizer.zero_grad()
         outputs = model(images)
         loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
         loss.backward()
         optimizer.step()
         toplam_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {toplam_loss/len(trainset)}")
    train_losses.append(toplam_loss/len(trainset))   

    validasyon_loss = 0
    with torch.no_grad():
        for images, labels in valset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validasyon_loss += loss.item()
    print(f"Validation Loss: {validasyon_loss/len(valset)}")
    val_losses.append(validasyon_loss/len(valset))

    if (validasyon_loss/len(valset)) < en_iyi_val_loss:
     en_iyi_val_loss = validasyon_loss/len(valset)
     torch.save(model.state_dict(), "model_efficientnet_best.pth")
     print(f"  → Yeni en iyi model kaydedildi (epoch {epoch+1})")

torch.save(model.state_dict(), "model_efficientnet.pth")
print("Model kaydedildi: model_efficientnet.pth")

tum_gercekler = []
tum_tahminler = []
with torch.no_grad():
    for images, labels in valset:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, tahminler = torch.max(outputs, 1)
        tum_gercekler.extend(labels.cpu().numpy())
        tum_tahminler.extend(tahminler.cpu().numpy())
    print(classification_report(tum_gercekler, tum_tahminler, target_names=train_dataset_aug.classes))
    print(confusion_matrix(tum_gercekler, tum_tahminler))




plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_grafigi_efficientnet.png")   # dosyaya kaydet, hocana göstermek için
plt.show()    

print("\n--- EN İYİ MODEL (Epoch 5) DEĞERLENDİRMESİ ---")
model.load_state_dict(torch.load("model_efficientnet_best.pth"))
model.eval()

tum_gercekler_best = []
tum_tahminler_best = []
with torch.no_grad():
    for images, labels in valset:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, tahminler = torch.max(outputs, 1)
        tum_gercekler_best.extend(labels.cpu().numpy())
        tum_tahminler_best.extend(tahminler.cpu().numpy())
    print(classification_report(tum_gercekler_best, tum_tahminler_best, target_names=train_dataset_aug.classes))
    print(confusion_matrix(tum_gercekler_best, tum_tahminler_best))