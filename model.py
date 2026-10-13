import torch.nn as nn
import torch.nn.functional as F
class EmotionCNN(nn.Module):
    def __init__(self):
        super ().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)# 3x3 boyutunda bir evrişim katmanı in channels=1 (gri tonlu görüntü) ve out_channels=16 (16 filtre) ile başlar. Bu, modelin ilk katmanıdır ve giriş görüntüsünden özellikleri çıkarmaya başlar.
        self.bn1 = nn.BatchNorm2d(16)   # Batch Normalization işine yarıyor, modelin daha hızlı ve stabil öğrenmesini sağlıyor.
        self.pool1 = nn.MaxPool2d(2, 2) # 2x2 boyutunda bir max pooling katmanı, bu katman, özellik haritalarını küçültür ve modelin daha az parametre ile çalışmasını sağlar. Bu, modelin daha hızlı ve daha az bellek kullanarak çalışmasını sağlar.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32) 
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)  
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()# Flatten katmanı, evrişim ve havuzlama katmanlarından sonra elde edilen çok boyutlu özellik haritalarını tek boyutlu bir vektöre dönüştürür. Bu, tam bağlantılı katmanlara (fully connected layers) geçiş için gereklidir.
        self.dropout = nn.Dropout(0.3)# Dropout katmanı, modelin aşırı öğrenmesini (overfitting) önlemek için kullanılır. Eğitim sırasında, belirli bir oranda (bu durumda %30) nöronları rastgele devre dışı bırakır. Bu, modelin daha genel ve sağlam hale gelmesine yardımcı olur.
        self.fc1 = nn.Linear(256 * 5 * 5, 7)# Fully connected katmanı, modelin son katmanıdır ve sınıflandırma görevini gerçekleştirir. Bu katman, 128 * 6 * 6 boyutundaki giriş vektörünü alır ve 7 boyutlu bir çıkış vektörü üretir (bu durumda 7 farklı duygu sınıfı için).

    def forward(self, x): #forward metodu, modelin ileri geçişini tanımlar. Bu metod, giriş verilerini alır ve modelin katmanlarından geçirerek çıkış üretir. Her bir katman, giriş verilerini işler ve bir sonraki katmana iletir. Bu metod, modelin nasıl çalıştığını ve verilerin nasıl işlendiğini belirler.
        x= self.conv1(x)
        x = self.bn1(x) 
        x=F.relu(x)
        x= self.pool1(x)
        x= self.conv2(x)
        x = self.bn2(x) 
        x=F.relu(x)
        x= self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x) 
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)        # ÖNCE conv4
        x = F.relu(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        x = self.flatten(x)     # SONRA flatten
        x = self.dropout(x)
        x = self.fc1(x)
        return x

#TODO: Test amaçlı
if __name__ == "__main__":
    import torch
    model = EmotionCNN()
    test_girdi = torch.randn(1, 1, 224, 224)
    cikti = model(test_girdi)
    print(cikti.shape)
