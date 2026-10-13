import numpy as np
import cv2

#TODO: OpenCV ile kamera görüntüsü yakalama ve gösterme

cap = cv2.VideoCapture(0)   # 0 = bilgisayarın varsayılan kamerası

# Hazır yüz tespit modelini yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()   # Kameradan bir kare oku
                              # ret: okuma başarılı mı (True/False), frame: görüntünün kendisi (NumPy array)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Griye çevir

    detected_faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5)
    detected_eyes_in_face = []  # Gözleri yüzlerin içinde tespit etmek için boş bir liste oluştur

    # Yüzleri dikdörtgen ile çiz
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Yüzün bulunduğu bölgeyi kes
        face_roi_color = frame[y:y+h, x:x+w]
        face=grey[y:y+h, x:x+w] 

        # Yüzün içindeki gözleri tespit et
        detected_eyes_in_face = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5) 

        # Gözleri daire ile çiz
        for (x, y, w, h) in detected_eyes_in_face:
            cv2.circle(face_roi_color, (x+w//2, y+h//2), min(w, h)//2, (0, 255, 0), 2)
    
    cv2.imshow("Kamera", frame)
    cv2.imshow("Gri Kamera", grey)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Kapatılıyor...")
        print(detected_faces)
        print(detected_eyes_in_face)
        print(f"Detected faces: {len(detected_faces)}, Detected eyes: {len(detected_eyes_in_face)}")
        print(frame.shape)
        print(grey.shape) 
        break

cap.release()
cv2.destroyAllWindows()



