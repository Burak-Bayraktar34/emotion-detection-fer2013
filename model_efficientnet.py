import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def create_efficientnet_model(num_classes=7):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    # Son bloğu (features'ın son katmanı) fine-tuning için çöz
    for param in model.features[-3].parameters():
        param.requires_grad = True

    # Classifier'ın son katmanını (Linear kısmını) 7 sınıfa göre değiştir
    model.classifier[1] = nn.Linear(1280, num_classes)

    return model

#if __name__ == "__main__":
    #model = create_efficientnet_model()
    #test_input = torch.randn(1, 3, 224, 224)   # dikkat: 3 kanal, 224x224
    #output = model(test_input)
    #print(output.shape)