from torchvision import transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def init_transform_dict(clip_image_size, gen_image_size):
    tsfm_dict = {
        'val': transforms.Compose([
            transforms.Resize(clip_image_size, interpolation=BICUBIC),
            transforms.CenterCrop(clip_image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]),
        'train': transforms.Compose([
            transforms.RandomResizedCrop(clip_image_size, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, saturation=0, hue=0),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]),
        'test': transforms.Compose([
            transforms.Resize(gen_image_size, interpolation=BICUBIC),
            transforms.CenterCrop(gen_image_size),
            transforms.ToTensor(),
        ]),
    }

    return tsfm_dict
