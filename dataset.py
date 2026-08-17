import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from config import Backbone
from PIL import Image
from torch import tensor

from vocabulary import Vocabulary


def create_img_transform(backbone_name: Backbone):
    if backbone_name == "efficientnet_b3":
        image_size = 300
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        interpolation = InterpolationMode.BICUBIC
    elif backbone_name == "clip_vit_b16":
        image_size = 224
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        interpolation = InterpolationMode.BICUBIC
    elif backbone_name in {"resnet_50", "vit_b16"}:
        image_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        interpolation = InterpolationMode.BILINEAR
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=interpolation,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    

# nhận dữ liệu thô và trả về data theo cặp image-caption_ids
class MSCOCODataset(Dataset):
    def __init__(self, images_dir, df, transform, vocab: Vocabulary, max_length):
        self.images_dir = images_dir
        self.df = df
        self.transform = transform # transform cho ảnh
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        # df.size = số dòng * sô cột
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.images_dir, row["filepath"], row["filename"])
        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        tokens = row["tokens"]
        input_ids, attention_mask = self.vocab.encode_from_tokens(tokens, self.max_length)

        input_ids = tensor(input_ids)
        attention_mask = tensor(attention_mask)

        return image, input_ids, attention_mask
