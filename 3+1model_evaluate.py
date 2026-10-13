import torch
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import EmotionCNN
from model_resnet import create_resnet_model
from model_efficientnet import create_efficientnet_model
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

test_yolu = "dataset/test"

# Kendi CNN'in için format: 1 kanal, 224x224, normalize yok
cnn_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

# ResNet50/EfficientNet için format: 3 kanal, 224x224, ImageNet normalize
pretrained_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Aynı klasörden, iki farklı transform ile iki dataset
cnn_dataset = ImageFolder(root=test_yolu, transform=cnn_transform)
pretrained_dataset = ImageFolder(root=test_yolu, transform=pretrained_transform)

# ÖNEMLİ: shuffle=False, ikisi de aynı sırada olmalı ki aynı index aynı görüntüye karşılık gelsin
cnn_loader = DataLoader(cnn_dataset, batch_size=32, shuffle=False)
pretrained_loader = DataLoader(pretrained_dataset, batch_size=32, shuffle=False)

# Üç modeli de yükle
cnn_model = EmotionCNN()
cnn_model.load_state_dict(torch.load("model.pth"))
cnn_model = cnn_model.to(device)
cnn_model.eval()

resnet_model = create_resnet_model(num_classes=7)
resnet_model.load_state_dict(torch.load("model_resnet.pth"))
resnet_model = resnet_model.to(device)
resnet_model.eval()

efficientnet_model = create_efficientnet_model(num_classes=7)
efficientnet_model.load_state_dict(torch.load("model_efficientnet.pth"))
efficientnet_model = efficientnet_model.to(device)
efficientnet_model.eval()

tum_gercekler = []
tum_tahminler = []

with torch.no_grad():
    # zip: iki loader'ı AYNI ANDA, batch batch gez (ikisi de aynı sırada olduğu için aynı görüntülere denk gelir)
    for (cnn_images, labels), (pretrained_images, _) in zip(cnn_loader, pretrained_loader):
        cnn_images = cnn_images.to(device)
        pretrained_images = pretrained_images.to(device)
        labels = labels.to(device)

        cnn_outputs = F.softmax(cnn_model(cnn_images), dim=1)
        resnet_outputs = F.softmax(resnet_model(pretrained_images), dim=1)
        efficientnet_outputs = F.softmax(efficientnet_model(pretrained_images), dim=1)

        # Üç modelin olasılıklarını ortala
        final_outputs = (cnn_outputs + resnet_outputs + efficientnet_outputs) / 3

        _, tahminler = torch.max(final_outputs, 1)
        tum_gercekler.extend(labels.cpu().numpy())
        tum_tahminler.extend(tahminler.cpu().numpy())

print("\n--- ENSEMBLE (3 MODEL BİRLEŞİMİ) — NİHAİ SONUÇ ---\n")
print(classification_report(tum_gercekler, tum_tahminler, target_names=cnn_dataset.classes))
print(confusion_matrix(tum_gercekler, tum_tahminler))


cm = confusion_matrix(tum_gercekler, tum_tahminler)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cnn_dataset.classes, yticklabels=cnn_dataset.classes)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Ensemble (3 Model Birleşimi) — Confusion Matrix')
plt.savefig("ensemble_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()