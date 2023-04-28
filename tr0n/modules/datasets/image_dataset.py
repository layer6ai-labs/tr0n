import os
from typing import Callable, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, root: str, transform: Callable, transform_orig: Callable) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.transform_orig = transform_orig
        
        files = os.listdir(self.root)
        self.img_paths = []
        for f in sorted(files):
            # make sure file is an image
            if f.endswith(('.jpg', '.png', 'jpeg')):
                img_path = os.path.join(self.root, f)
                self.img_paths.append(img_path)

    def _load_image(self, idx: int) -> Image.Image:
        img_path = self.img_paths[idx]
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._load_image(idx)
        image_orig = self.transform_orig(image)
        image = self.transform(image)
        return image_orig, image

    def __len__(self) -> int:
        return len(self.img_paths)
