"""Proposal bindings: retain P1 models, replace core input geometry only.

Use this module instead of reproduction.p1.models for the three core methods.
Historical P1 runners and their accepted outputs remain unchanged.
"""
import io
import random

from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as T

from reproduction.p1.models import Images, MEAN, STD, build, loss, scores

CORE_METHODS = ('frozen_L14_resize', 'effort', 'effort_degradation')
PREPROCESS_VERSION = 'proposal_minshort224_bicubic_crop_v1'


def resize_small_crop(img, train, rng, size=224):
    """Upscale only undersized images; never add padding or distort aspect ratio."""
    w, h = img.size
    if min(w, h) < size:
        scale = size / min(w, h)
        w, h = max(size, round(w * scale)), max(size, round(h * scale))
        img = img.resize((w, h), Image.Resampling.BICUBIC)
    if train:
        x, y = rng.randint(0, w - size), rng.randint(0, h - size)
    else:
        x, y = round((w - size) / 2), round((h - size) / 2)
    return img.crop((x, y, x + size, y + size))


def degrade(img, seed):
    """Unchanged P1 probabilities, strengths, interpolation and operation order."""
    op = random.Random(seed + 2000).randrange(10)
    if 1 <= op <= 3:
        with io.BytesIO() as buffer:
            img.save(buffer, 'JPEG', quality=[90, 70, 50][op - 1], subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                img = decoded.convert('RGB')
    elif 4 <= op <= 6:
        img = img.filter(ImageFilter.GaussianBlur([.5, 1, 2][op - 4]))
    elif op >= 7:
        w, h = img.size
        scale = [.75, .5, .25][op - 7]
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.Resampling.BICUBIC).resize((w, h), Image.Resampling.BICUBIC)
    return img


class Transform:
    def __init__(self, method, train, meta):
        if method not in CORE_METHODS:
            raise ValueError(f'Proposal core transform does not support {method}')
        self.method, self.train = method, train
        self.norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __call__(self, img, seed):
        if self.train and self.method == 'effort_degradation':
            img = degrade(img, seed)
        rng = random.Random(seed)
        if self.train and rng.random() < .5:
            img = ImageOps.mirror(img)
        return self.norm(resize_small_crop(img, self.train, rng))
