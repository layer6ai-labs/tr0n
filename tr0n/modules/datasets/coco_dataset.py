import os.path
import random
from typing import Any, Callable, Optional, Tuple, List
from PIL import Image
from torchvision.datasets import VisionDataset

class CocoCaptions(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.train = train

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        annotations = self.coco.loadAnns(self.coco.getAnnIds(id))
        return [ann["caption"] for ann in annotations]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        targets = self._load_target(id)
        if self.train:
            random_target_idx = random.randrange(len(targets))
            target = targets[random_target_idx]
        else:
            target = targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def __len__(self) -> int:
        return len(self.ids)
