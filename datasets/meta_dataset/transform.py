import torch
import random
import torchvision.transforms as transforms
from PIL import ImageEnhance

from .utils import Split
from .config import DataConfig

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)


class ImageJitter(object):
    def __init__(self, transformdict):
        transformtypedict = dict(
            Brightness=ImageEnhance.Brightness,
            Contrast=ImageEnhance.Contrast,
            Sharpness=ImageEnhance.Sharpness,
            Color=ImageEnhance.Color,
        )
        self.params = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.params))

        for i, (transformer, alpha) in enumerate(self.params):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out


def get_transforms(data_config: DataConfig, split, transform_type: Split):
    if transform_type == "no_transform":
        return basic_transform(data_config, split)
    else:
        if transform_type == Split["TRAIN"]:
            return train_transform(data_config)
        else:
            return test_transform(data_config)


def test_transform(data_config: DataConfig):
    resize_size = int(data_config.image_size * 256 / 224)
    assert resize_size == data_config.image_size * 256 // 224
    # resize_size = data_config.image_size

    transf_dict = {
        "resize": transforms.Resize(resize_size),
        "center_crop": transforms.CenterCrop(data_config.image_size),
        "to_tensor": transforms.ToTensor(),
        # "normalize": normalize,
    }
    augmentations = data_config.test_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])


def train_transform(data_config: DataConfig):
    transf_dict = {
        "resize": transforms.Resize(data_config.image_size),
        "center_crop": transforms.CenterCrop(data_config.image_size),
        "random_resized_crop": transforms.RandomResizedCrop(data_config.image_size),
        "jitter": ImageJitter(jitter_param),
        "random_flip": transforms.RandomHorizontalFlip(),
        "to_tensor": transforms.ToTensor(),
        # "normalize": normalize,
    }
    augmentations = data_config.train_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])


def basic_transform(data_config: DataConfig, split: Split):
    if split == Split["TRAIN"]:
        resize_size = data_config.image_size
    else:
        resize_size = int(data_config.image_size * 256 / 224)
        resize_size = (resize_size, resize_size)
    transf_dict = {
        "resize": transforms.Resize(resize_size),
        "to_tensor": transforms.ToTensor(),
    }
    augmentations = list(transf_dict.keys())

    return transforms.Compose([transf_dict[key] for key in augmentations])


def to_tensor(data_config: DataConfig):
    transf_dict = {
        # "to_tensor": transforms.ToTensor(),
        "normalize": normalize,
    }
    augmentations = list(transf_dict.keys())

    return transforms.Compose([transf_dict[key] for key in augmentations])


def random_augmentation(data_config: DataConfig, split: Split):
    _jitter_param = dict(Brightness=1.0, Contrast=1.0, Color=1.0)
    if split == Split["TRAIN"]:
        resize_size = data_config.image_size
    else:
        resize_size = int(data_config.image_size * 256 / 224)
    augmentation_dict = {
        "crop_and_resize": {
            "to_image": transforms.ToPILImage(),
            "resize": transforms.Resize(resize_size),
            "random_resized_crop": transforms.RandomResizedCrop(data_config.image_size),
            "flip": transforms.RandomHorizontalFlip(),
            "to_tensor": transforms.ToTensor(),
            "normalize": normalize,
        },
        "distort(drop)": {
            "to_image": transforms.ToPILImage(),
            "resize": transforms.Resize(resize_size),
            "jitter": ImageJitter(_jitter_param),
            "to_tensor": transforms.ToTensor(),
            "normalize": normalize,
        },
        "blur": {
            "to_image": transforms.ToPILImage(),
            "resize": transforms.Resize(resize_size),
            "gaussian_blur": transforms.GaussianBlur(3),
            "to_tensor": transforms.ToTensor(),
            "normalize": normalize,
        },
    }
    # Randomly choose one augmentation
    transf_dict = augmentation_dict[random.choice(list(augmentation_dict.keys()))]
    augmentations = list(transf_dict.keys())

    return transforms.Compose([transf_dict[key] for key in augmentations])
