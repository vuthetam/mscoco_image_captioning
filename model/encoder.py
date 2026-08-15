import torch
from torch import Tensor, nn
from torchvision import models
from transformers import CLIPVisionModel
from config import Backbone


class ResNet50Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        vit = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # bỏ lớp AvgPool và FC
        custom = list(vit.children())[:-2] 

        self.backbone = nn.Sequential(*custom)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # output feature cuối của resnet50 là [b, 2048, 7, 7]
        # số channels = số kernel, mục tiêu nén xuống còn bằng d_model
        self.conv1x1 = nn.Conv2d(in_channels=2048, out_channels=d_model, kernel_size=1)

    def forward(self, images):
        # backbone sẽ không update BatchNorm stats
        self.backbone.eval()
        with torch.no_grad():
            features: Tensor = self.backbone(images)
        features = self.conv1x1(features) # [B, d_model, 7, 7]
        features = features.flatten(2) # nén chiều 2,3 thành 1. [B, d_model, 49]
        features = features.permute(0, 2, 1)
        return features


class EfficientNetB3Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

        self.backbone = efficientnet.features
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.conv1x1 = nn.Conv2d(1536, d_model, kernel_size=1)

    def forward(self, images):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(images)   # [B, 3, 224, 224] -> [B, 768, H', W']
        features = self.conv1x1(features)  # [B, 1536, H, W] -> [B, d_model, H, W]
        features = features.flatten(2).permute(0,2,1)
        return features


class CLIPViTB16Encoder(nn.Module):
    MODEL_NAME = "openai/clip-vit-base-patch16"

    def __init__(self, d_model):
        super().__init__()
        clip_model = CLIPVisionModel.from_pretrained(self.MODEL_NAME)
        self.backbone = getattr(clip_model, "vision_model", clip_model)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(self.backbone.config.hidden_size, d_model)

    def forward(self, images):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(pixel_values=images).last_hidden_state

        # CLIP ViT-B/16 returns one CLS token and 14x14 patch tokens.
        return self.projection(features)  # [B, 197, 768] -> [B, 197, d_model]



def create_encoder(backbone_name: Backbone, d_model):
    if backbone_name == "resnet_50":
        return ResNet50Encoder(d_model)
    elif backbone_name == "efficientnet_b3":
        return EfficientNetB3Encoder(d_model)
    elif backbone_name == "clip_vit_b16":
        return CLIPViTB16Encoder(d_model)
    else:
        raise ValueError(f"Unsupported backbone {backbone_name}")
    
