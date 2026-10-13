import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model_efficientnet import create_efficientnet_model
from sklearn.metrics import classification_report, confusion_matrix
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")
 
test_yolu = "dataset/test"
 
# Test için kullanılan sade transform (train_resnet.py'daki val_transform ile birebir aynı)
val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
 
test_dataset = ImageFolder(root=test_yolu, transform=val_transform)
testset = DataLoader(test_dataset, batch_size=16, shuffle=False)
 
print(len(test_dataset))
print(test_dataset.classes)
 
# Modeli oluştur ve nihai (son epoch) ağırlıkları yükle
model = create_efficientnet_model(num_classes=7)
model.load_state_dict(torch.load("model_efficientnet.pth"))
model = model.to(device)
model.eval()
 
tum_gercekler = []
tum_tahminler = []
with torch.no_grad():
    for images, labels in testset:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, tahminler = torch.max(outputs, 1)
        tum_gercekler.extend(labels.cpu().numpy())
        tum_tahminler.extend(tahminler.cpu().numpy())
 
print("\n--- EfficientNet NİHAİ MODEL — TEST SETİ DEĞERLENDİRMESİ ---\n")
print(classification_report(tum_gercekler, tum_tahminler, target_names=test_dataset.classes))
print(confusion_matrix(tum_gercekler, tum_tahminler))
 