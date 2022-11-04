import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision import models

from models.DeepCrack.CrackNet import DeepCrackNet, init_net
from models.TransUNet.vit_seg_modeling import get_transunet


class FCNRes101(nn.Module):
    def __init__(self, pretrained=True):
        super(FCNRes101, self).__init__()
        self.model = models.segmentation.fcn_resnet101(pretrained=pretrained)

        # output class
        self.model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
    
    def forward(self, x):
        return self.model(x)["out"]


class DeepLabV3_Res101(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepLabV3_Res101, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)

        # output class
        self.model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]


class UNet_PlusPlus(nn.Module):
    def __init__(self, encoder='efficientnet-b4', pretrained=True):
        super(UNet_PlusPlus, self).__init__()
        if pretrained == True:
            weights='imagenet'
        else:
            weights=None
        self.model = smp.UnetPlusPlus(encoder_name=encoder,
                                      encoder_weights=weights,
                                      in_channels=3,
                                      classes=2)

    def forward(self, x):
        return self.model(x)


class TransUnet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(TransUnet, self).__init__()
        self.backbone = get_transunet(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.backbone(x)


class DeepCrack(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepCrack, self).__init__()
        self.model = DeepCrackNet(3, 2, 64, norm='batch')
        self.model = init_net(self.model, 'xavier', 0.02, [0])

        if pretrained:
            checkpoint = torch.load('weights/pretrained_net_G.pth')

            for k in list(checkpoint.keys()):
                name = k.replace('module', '')
                checkpoint[name] = checkpoint.pop(k)
            checkpoint = checkpoint
            model_state_dict = self.model.state_dict()

            for k in model_state_dict.keys():
                if k not in checkpoint:
                    raise Exception('model state dict load key error')
                elif model_state_dict[k].size() == checkpoint[k].size():
                    model_state_dict[k] = checkpoint[k]
                else:
                    print(f"model state dict load skip {k}")
            self.model.load_state_dict(model_state_dict)


    def forward(self, x):
        return self.model(x)[-1]
