import random

import numpy as np
from imgaug import augmenters as iaa

from dataloaders.augmentations.composition import Compose, OneOf
import dataloaders.augmentations.functional as F


def to_tuple(param, low=None):
    if isinstance(param, tuple):
        return param
    else:
        return (-param if low is None else low, param)


class BasicTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, **kwargs):
        if random.random() < self.prob:
            params = self.get_params()
            return {
                k: self.apply(a, **params) if k in self.targets else a 
                for k, a in kwargs.items()
            }
        return kwargs

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        raise NotImplementedError


class BasicIAATransform(BasicTransform):
    def __init__(self, prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Noop()
        self.deterministic_processor = iaa.Noop()

    def __call__(self, **kwargs):
        self.deterministic_processor = self.processor.to_deterministic()
        return super().__call__(**kwargs)

    def apply(self, img, **params):
        return self.deterministic_processor.augment_image(img)


class DualTransform(BasicTransform):
    """
    transfrom for segmentation task
    """
    @property
    def targets(self):
        return 'image', 'mask'


class DualIAATransform(DualTransform, BasicIAATransform):
    pass


class ImageOnlyTransform(BasicTransform):
    """
    transforms applied to image only
    """
    @property
    def targets(self):
        return 'image'


class ImageOnlyIAATransform(ImageOnlyTransform, BasicIAATransform):
    pass


class RandomCrop:
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width

    def apply(self, img, mask):
        assert img.shape[0] >= self.height
        assert img.shape[1] >= self.width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = random.randint(0, img.shape[1] - self.width)
        y = random.randint(0, img.shape[0] - self.height)
        img = img[y:y+self.height, x:x+self.width]
        mask = mask[y:y+self.height, x:x+self.width]
        return img, mask


class Flip(DualTransform):
    def apply(self, img, d=0):
        return F.random_flip(img, d)

    def get_params(self):
        return {'d': random.randint(-1, 1)}


class Transpose(DualTransform):
    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    def apply(self, img, factor=0):
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        return {'factor': random.randint(0, 4)}


class ShiftScaleRotate(DualTransform):
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, 
                 rotate_limit=45, prob=0.5):
        super().__init__(prob)
        self.shift_limit = to_tuple(shift_limit)
        self.scale_limit = to_tuple(scale_limit)
        self.rotate_limit = to_tuple(rotate_limit)

    def apply(self, img, angle=0, scale=0, dx=0, dy=0):
        return F.shift_scale_rotate(img, angle, scale, dx, dy)

    def get_params(self):
        return {
            'angle': random.uniform(self.rotate_limit[0],
                                    self.rotate_limit[1]),
            'scale': random.uniform(1+self.scale_limit[0],
                                    1+self.scale_limit[1]),
            'dx': round(random.uniform(self.shift_limit[0],
                                       self.shift_limit[1])),
            'dy': round(random.uniform(self.shift_limit[0],
                                       self.shift_limit[1]))
        }


class Distort1(DualTransform):
    def __init__(self, distort_limit=0.05, shift_limit=0.05, prob=0.5):
        super().__init__(prob)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.shift_limit = to_tuple(shift_limit)

    def apply(self, img, k=0, dx=0, dy=0):
        return F.distort1(img, k, dx, dy)

    def get_params(self):
        return {
            'k': random.uniform(self.distort_limit[0], 
                                self.distort_limit[1]),
            'dx': round(random.uniform(self.shift_limit[0], 
                                       self.shift_limit[1])),
            'dy': round(random.uniform(self.shift_limit[0], 
                                       self.shift_limit[1]))
        }


class Distort2(DualTransform):
    def __init__(self, num_steps=5, distort_limit=0.3, prob=0.5):
        super().__init__(prob)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.prob = prob

    def apply(self, img, stepsx=[], stepsy=[]):
        return F.distort2(img, self.num_steps, stepsx, stepsy)

    def get_params(self):
        stepsx = [1 + random.uniform(self.distort_limit[0], 
                                     self.distort_limit[1]) 
                  for i in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], 
                                     self.distort_limit[1]) 
                  for i in range(self.num_steps + 1)]
        return {
            'stepsx': stepsx,
            'stepsy': stepsy
        }


class ElasticTransform(DualTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, prob=0.5):
        super().__init__(prob)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma

    def apply(self, img, random_state=None):
        return F.elastic_transform_fast(
            img, self.alpha, self.sigma, self.alpha_affine, 
            np.random.RandomState(random_state))

    def get_params(self):
        return {'random_state': np.random.randint(0, 10000)}


class RandomBrightness(ImageOnlyTransform):
    def __init__(self, limit=0.2, prob=0.5):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2):
        return F.random_brightness(img, alpha)

    def get_params(self):
        return {'alpha': 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class RandomContrast(ImageOnlyTransform):
    def __init__(self, limit=0.2, prob=.5):
        super().__init__(prob)
        self.limit = to_tuple(limit)

    def apply(self, img, alpha=0.2):
        return F.random_contrast(img, alpha)

    def get_params(self):
        return {'alpha': 1.0 + np.random.uniform(self.limit[0], self.limit[1])}


class Blur(ImageOnlyTransform):
    def __init__(self, blur_limit=7, prob=.5):
        super().__init__(prob)
        self.blur_limit = to_tuple(blur_limit, 3)

    def apply(self, image, ksize=3):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            'ksize': np.random.choice(np.arange(self.blur_limit[0], 
                                                self.blur_limit[1] + 1, 2))
        }


class MotionBlur(Blur):
    def apply(self, img, ksize=9):
        return F.motion_blur(img, ksize=ksize)


class MedianBlur(Blur):
    def apply(self, image, ksize=3):
        return F.median_blur(image, ksize)


class GaussNoise(ImageOnlyTransform):
    def __init__(self, var_limit=(10, 50), prob=.5):
        super().__init__(prob)
        self.var_limit = to_tuple(var_limit)

    def apply(self, img, var=30):
        return F.gauss_noise(img, var=var)

    def get_params(self):
        return {
            'var': np.random.randint(self.var_limit[0], self.var_limit[1])
        }


class IAASharpen(ImageOnlyIAATransform):
    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.Sharpen(alpha, lightness)


class IAAAdditiveGaussianNoise(ImageOnlyIAATransform):
    def __init__(self, loc=0, scale=(0.01*255, 0.05*255), prob=0.5):
        super().__init__(prob)
        self.processor = iaa.AdditiveGaussianNoise(loc, scale)


class IAAPiecewiseAffine(DualIAATransform):
    def __init__(self, scale=(0.03, 0.05), nb_rows=4, nb_cols=4, prob=.5):
        super().__init__(prob)
        self.processor = iaa.PiecewiseAffine(scale, nb_rows, nb_cols)


class IAAPerspective(DualIAATransform):
    def __init__(self, scale=(0.05, 0.1), prob=.5):
        super().__init__(prob)
        self.processor = iaa.PerspectiveTransform(scale)


def aug_random(prob=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(prob=0.5),
            GaussNoise(prob=0.5),
        ], prob=0.2),
        OneOf([
            MotionBlur(prob=0.2),
            MedianBlur(blur_limit=3, prob=0.3),
            Blur(blur_limit=3, prob=0.5),
        ], prob=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                         rotate_limit=45, prob=0.2),
        OneOf([
            Distort1(prob=0.2),
            Distort2(prob=0.2),
            IAAPiecewiseAffine(prob=0.2),
            IAAPerspective(prob=0.2),
            ElasticTransform(prob=0.2)
        ], prob=0.3),
        OneOf([
            IAASharpen(),
            RandomContrast(),
            RandomBrightness(),
        ], prob=0.3)
    ])

