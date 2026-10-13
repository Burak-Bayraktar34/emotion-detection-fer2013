import torch
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from model_resnet import create_resnet_model
from model_efficientnet import create_efficientnet_model
import torch.nn.functional as f

from model import EmotionCNN


duygular = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Görüntü Yakalama")

# Bir kere kurulacak şeyler (kamera + modeller) - fonksiyon DIŞINDA
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)#Gecikme önleyici bir işe yaramadı ama

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

ilk_frame = True
frame_sayaci = 0
son_tahmin = "..."

cnn_model = EmotionCNN()
cnn_model.load_state_dict(torch.load("model.pth"))
cnn_model.eval()

resnet_model = create_resnet_model(num_classes=7)
resnet_model.load_state_dict(torch.load("model_resnet.pth"))
resnet_model.eval()

efficientnet_model = create_efficientnet_model(num_classes=7)
efficientnet_model.load_state_dict(torch.load("model_efficientnet.pth"))
efficientnet_model.eval()

imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

cnn_model = cnn_model.to(device)
resnet_model = resnet_model.to(device)
efficientnet_model = efficientnet_model.to(device)

imagenet_mean = imagenet_mean.to(device)
imagenet_std = imagenet_std.to(device)

def guncelleme():
    global ilk_frame, frame_sayaci, son_tahmin
    frame_sayaci += 1
    ret, frame = cam.read()
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_gray = gri[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        resize_face_gray = cv2.resize(face_gray, (224, 224))
        #convert to tensor and normalize
        cnn_input_np = resize_face_gray / 255.0

        cnn_input = torch.tensor(cnn_input_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        cnn_input = cnn_input.to(device)
        # --- Pretrained (ResNet/EfficientNet) formatı ---
        face_3kanal = cv2.cvtColor(resize_face_gray, cv2.COLOR_GRAY2RGB)  # 1 kanaldan 3 kanala kopyala
        pretrained_input_np = face_3kanal / 255.0
        pretrained_input = torch.tensor(pretrained_input_np, dtype=torch.float32).permute(2,0,1)
        pretrained_input = pretrained_input.to(device)  # HWC -> CHW
        pretrained_input = (pretrained_input - imagenet_mean) / imagenet_std
        pretrained_input = pretrained_input.unsqueeze(0)
        
        if frame_sayaci % 5 == 0:
            with torch.no_grad():
                p1 = f.softmax(cnn_model(cnn_input), dim=1)
                p2 = f.softmax(resnet_model(pretrained_input), dim=1)
                p3 = f.softmax(efficientnet_model(pretrained_input), dim=1)
                final_probs = (p1 + p2 + p3) / 3
                predicted_class = torch.argmax(final_probs, dim=1).item()
                son_tahmin = duygular[predicted_class]

        detected_eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.2, minNeighbors=7)
        for (ex, ey, ew, eh) in detected_eyes:
             cv2.circle(face_color, (ex+ew//2, ey+eh//2), min(ew, eh)//2, (0, 255, 0), 2)
        cv2.putText(frame, son_tahmin, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2) 

    # RGB yapma 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)

    label.setPixmap(pixmap)
    label.adjustSize()

    if ilk_frame:
        window.adjustSize()
        ilk_frame = False

label = QLabel(window)
label.move(100, 50)


timer = QTimer()
timer.timeout.connect(guncelleme)
timer.start(30)

window.show()
sys.exit(app.exec_())