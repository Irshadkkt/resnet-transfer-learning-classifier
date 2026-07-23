import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet normalization (expected by pretrained ResNet models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class ChristmasImages(Dataset):
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path = path

        if self.training:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

            self.samples = []
            self.class_to_idx = {}

            classes = sorted(os.listdir(path))
            for idx, cls in enumerate(classes):
                class_path = os.path.join(path, cls)
                if not os.path.isdir(class_path):
                    continue

                self.class_to_idx[cls] = idx

                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, idx))
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

            files = [
                f for f in os.listdir(path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))  # 0.png -> 0
            self.samples = [os.path.join(path, f) for f in files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.training:
            img_path, label = self.samples[index]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label
        else:
            img_path = self.samples[index]
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return (image,)
