import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet_model(num_classes=7):
    # Load the pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze all layers except the final fully connected layer  
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer to match the number of classes (7 for emotions)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

#if __name__ == "__main__":
    #model = create_resnet_model()
    #test_girdi = torch.randn(1, 3, 224, 224)   # dikkat: 3 kanal, 224x224
    #cikti = model(test_girdi)
    #print(cikti.shape)