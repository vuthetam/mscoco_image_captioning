import torch
from torch import Tensor, nn
from torchvision import models
from transformers import CLIPVisionModel
from config import Backbone


class ResNet50Encoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove global average pooling and the classification layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.requires_grad_(False)

        # Convert each 2048-dimensional spatial feature to d_model.
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        # Keep BatchNorm statistics fixed while using the frozen backbone.
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(images)  # [B, 2048, 7, 7]

        # Treat each position in the 7x7 feature map as one image token.
        return self.projection(features).flatten(2).transpose(1, 2)


class EfficientNetB3Encoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        # .features excludes global pooling and the classification layer.
        self.backbone = efficientnet.features
        self.backbone.requires_grad_(False)

        self.projection = nn.Conv2d(1536, d_model, kernel_size=1)

    def forward(self, images: Tensor) -> Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(images)  # [B, 1536, 10, 10]

        # Convert the 10x10 feature map into 100 image tokens.
        return self.projection(features).flatten(2).transpose(1, 2)


class ViTB16Encoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.backbone.requires_grad_(False)

        self.projection = nn.Linear(self.backbone.hidden_dim, d_model)

    def forward(self, images):
        self.backbone.eval()
        with torch.no_grad():
            # Convert each image into 14x14 patch embeddings.
            features = self.backbone._process_input(images)

            # The pretrained encoder expects CLS together with patch tokens.
            class_token = self.backbone.class_token.expand(images.size(0), -1, -1)
            features = torch.cat([class_token, features], dim=1)
            features = self.backbone.encoder(features)

        # Remove CLS so the decoder receives only 196 spatial patch tokens.
        features = features[:, 1:, :]
        return self.projection(features)  # [B, 196, 768] -> [B, 196, d_model]


class CLIPViTB16Encoder(nn.Module):
    MODEL_NAME = "openai/clip-vit-base-patch16"

    def __init__(self, d_model):
        super().__init__()
        clip_model = CLIPVisionModel.from_pretrained(self.MODEL_NAME)

        # Support versions that wrap the actual backbone in .vision_model.
        self.backbone = getattr(clip_model, "vision_model", clip_model)
        self.backbone.requires_grad_(False)

        self.projection = nn.Linear(self.backbone.config.hidden_size, d_model)

    def forward(self, images):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(pixel_values=images).last_hidden_state

        # Remove CLS so the decoder receives only 196 spatial patch tokens.
        features = features[:, 1:, :]
        return self.projection(features)  # [B, 196, 768] -> [B, 196, d_model]



def create_encoder(backbone_name: Backbone, d_model):
    if backbone_name == "resnet_50":
        return ResNet50Encoder(d_model)
    elif backbone_name == "efficientnet_b3":
        return EfficientNetB3Encoder(d_model)
    elif backbone_name == "vit_b16":
        return ViTB16Encoder(d_model)
    elif backbone_name == "clip_vit_b16":
        return CLIPViTB16Encoder(d_model)
    else:
        raise ValueError(f"Unsupported backbone {backbone_name}")
    
