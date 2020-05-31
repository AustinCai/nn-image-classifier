# code in this file adapted from ildoonet/pytorch-randaugment
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image

# helpers for train_model.py and train_gmaxup.py ========================================================
# =======================================================================================================

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def SoftEqualize(img, _):
    # print("SoftEqualize")

    img_np = np.array(img).astype(np.int)
    equal_img = PIL.ImageOps.equalize(img)
    equal_img_np = np.array(equal_img).astype(np.int)
    # print(type(img_np))
    # print(img_np.shape)
    # print(type(equal_img_np))
    # print(equal_img_np.shape)
    soft_equal_img_np = np.divide(np.add(img_np, equal_img_np), 2.0).astype(np.uint8)
    # print(type(soft_equal_img_np))
    # print(soft_equal_img_np.shape)
    return Image.fromarray(soft_equal_img_np)

def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list(magnitude_range="orig"):  # 16 oeprations and their ranges
    
    bot = 0.6
    top = 1.4

    lists = {
        'orig': [
            (AutoContrast, 0, 1, "AutoContrast"),
            (Equalize, 0, 1, "Equalize"),
            (Invert, 0, 1, "Invert"),
            (Rotate, 0, 30, "Rotate"),
            (Posterize, 0, 4, "Posterize"),
            (Solarize, 0, 256, "Solarize"),
            (SolarizeAdd, 0, 110, "SolarizeAdd"),
            (Color, 0.1, 1.9, "Color"),
            (Contrast, 0.1, 1.9, "Contrast"),
            (Brightness, 0.1, 1.9, "Brightness"),
            (Sharpness, 0.1, 1.9, "Sharpness"),
            (ShearX, 0., 0.3, "ShearX"),
            (ShearY, 0., 0.3, "ShearY"),
            (CutoutAbs, 0, 40, "CutoutAbs"),
            (TranslateXabs, 0., 100, "TranslateXabs"),
            (TranslateYabs, 0., 100, "TranslateYabs")
        ],

        'reduced': [
            (AutoContrast, 0, 1, "AutoContrast"),
            (Equalize, 0, 1, "Equalize"),
            # (Invert, 0, 1),
            (Rotate, 0, 30, "Rotate"),
            (Posterize, 2, 4, "Posterize"),
            (Solarize, 128, 256, "SolarizeTop"),
            (Solarize, 0, 128, "SolarizeBot"),
            # (SolarizeAdd, 0, 28),
            (Color, 0.5, 1.5, "Color"),
            (Contrast, 0.5, 1.5, "Contrast"),
            (Brightness, 0.5, 1.5, "Brightness"),
            (Sharpness, 0.5, 1.5, "Sharpness"),
            (ShearX, 0., 0.3, "ShearX"),
            (ShearY, 0., 0.3, "ShearY"),
            (CutoutAbs, 0, 20, "CutoutAbs"),
            (TranslateXabs, 0., 70, "TranslateXabs"),
            (TranslateYabs, 0., 70, "TranslateYabs"),
        ],

        'deterministic': [
            # (AutoContrast, 0, 1, "AutoContrast"),
            (SoftEqualize, 0, 1, "SoftEqualize"),
            # (Invert, 0, 1),
            (Rotate, 30, 30, "Rotate"),
            (Posterize, 3, 3, "Posterize"),
            (Solarize, 192, 192, "Solarize") if random.randint(0, 1) < 1 else (Solarize, 64, 64, "Solarize"),
            # (SolarizeAdd, 0, 28),
            (Color, top, top, "Color") if random.randint(0, 1) < 1 else (Color, bot, bot, "Color"),
            (Contrast, top, top, "Contrast") if random.randint(0, 1) < 1 else (Contrast, bot, bot, "Contrast"),
            (Brightness, 1.3, 1.3, "Brightness") if random.randint(0, 1) < 1 else (Brightness, 0.7, 0.7, "Brightness"),
            (Sharpness, top, top, "Sharpness") if random.randint(0, 1) < 1 else (Sharpness, bot, bot, "Sharpness"),
            (ShearX, 0.2, 0.2, "ShearX"),
            (ShearY, 0.2, 0.2, "ShearY"),
            (CutoutAbs, 7, 7, "CutoutAbs"),
            (TranslateXabs, 7, 7, "TranslateXabs"),
            (TranslateYabs, 7, 7, "TranslateYabs"),
        ]

    }

    return lists[magnitude_range]


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


# helpers for train_model.py ============================================================================
# =======================================================================================================


# helpers for train_gmaxup.py ===========================================================================
# =======================================================================================================


# helpers for testing ===================================================================================
# =======================================================================================================

def test():
	pass


if __name__ == "__main__":
    test()