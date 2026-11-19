"""Identical preprocessing and perturbations for real and generated images."""

import io
import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_CONDITIONS = ("clean", "jpeg:90", "jpeg:70", "jpeg:50",
                      "resize:0.5", "resize:0.25", "blur:1", "blur:2")
PAIRED_CONDITIONS = ("clean", "jpeg:70", "resize:0.5",
                     "resize:0.5>jpeg:70", "jpeg:70>resize:0.5")


def parse_condition(condition):
    if ">" in condition:
        steps = condition.split(">")
        if not 2 <= len(steps) <= 4 or any(not step for step in steps):
            raise ValueError(f"Invalid processing chain: {condition}")
        for step in steps:
            parse_condition(step)
        return "chain", steps
    if condition == "clean":
        return "clean", None
    try:
        name, number = condition.split(":")
        value = float(number)
    except ValueError as error:
        raise ValueError(f"Invalid condition {condition!r}; use jpeg:70 / resize:0.5 / blur:1") from error
    valid = (math.isfinite(value) and (
        (name == "jpeg" and value.is_integer() and 1 <= value <= 100)
        or (name == "resize" and 0 < value < 1)
        or (name == "blur" and 0 < value <= 20)))
    if not valid:
        raise ValueError(f"Invalid condition: {condition}")
    return name, value


def perturb(image, condition):
    name, value = parse_condition(condition)
    if name == "chain":
        for step in value:
            image = perturb(image, step)
        return image
    if name == "clean":
        return image
    if name == "jpeg":
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=int(value), subsampling=2,
                   optimize=False, progressive=False)
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert("RGB")
    if name == "resize":
        small = (max(1, round(image.width * value)), max(1, round(image.height * value)))
        return image.resize(small, Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)
    return image.filter(ImageFilter.GaussianBlur(radius=value))


class ImageTransform:
    def __init__(self, image_size=224, training=False, augmentation="none", condition="clean"):
        parse_condition(condition)
        if augmentation not in ("none", "robust", "jpeg", "resize", "blur"):
            raise ValueError(f"Unknown augmentation: {augmentation}")
        self.training = training
        self.augmentation = augmentation
        self.condition = condition
        # Use the same fixed input dimensions for every source. Native resolution
        # and previous compression may STILL leave detectable source biases.
        self.tensor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if self.augmentation in ("robust", "jpeg") and random.random() < 0.5:
                image = perturb(image, f"jpeg:{random.randint(50, 100)}")
            if self.augmentation in ("robust", "resize") and random.random() < 0.5:
                image = perturb(image, f"resize:{random.uniform(0.25, 0.99)}")
            if self.augmentation in ("robust", "blur") and random.random() < 0.3:
                image = perturb(image, f"blur:{random.uniform(0.1, 2.0)}")
        return self.tensor(perturb(image, self.condition))


class ManifestDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root = Path(root).resolve()
        self.rows = rows
        self.transform = transform
        for row in rows:
            path = (self.root / row["path"]).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError(f"Missing/out-of-root image: {row['path']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), torch.tensor(float(row["label"])), row["path"], row["generator"]


class PairedImageTransform:
    """Independent views with one frozen law shared by B and C, both labels.

    Each view: independent horizontal flip (p=.5), uniform choice among the
    five development conditions, then the same resize/normalization as A.
    """
    def __init__(self, image_size=224):
        self.tensor = ImageTransform(image_size).tensor

    def __call__(self, image):
        result = []
        for _ in range(2):
            view = ImageOps.mirror(image) if random.random() < .5 else image
            view = perturb(view, random.choice(PAIRED_CONDITIONS))
            result.append(self.tensor(view))
        return tuple(result)
