import torch
import torchvision.models as models
def get_pretrained_resnet(name, pretrained=True):
    if name.lower() == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    
    return model