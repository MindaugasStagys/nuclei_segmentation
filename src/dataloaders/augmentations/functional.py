from scipy.ndimage.filters import gaussian_filter
from functools import wraps
import torchvision.transforms.functional as F
import numpy as np
import random
import torch
import math
import cv2


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def random_flip(img, code):
    return cv2.flip(img, code)


def transpose(img):
    return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)


def rot90(img, factor):
    img = np.rot90(img, factor)
    return np.ascontiguousarray(img)


def shift_scale_rotate(img, angle, scale, dx, dy):
    height, width = img.shape[:2]

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2 + dx*width, 
                                                     height/2 + dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height), 
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    return img


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype, maxval = img.dtype, np.max(img)
        return clip(func(img, *args, **kwargs), dtype, maxval)
    return wrapped_function


def blur(img, ksize):
    return cv2.blur(img, (ksize, ksize))


def median_blur(img, ksize):
    return cv2.medianBlur(img, ksize)


def motion_blur(img, ksize):
    kernel = np.zeros((ksize, ksize))
    xs, ys = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
    xe, ye = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
    cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
    return cv2.filter2D(img, -1, kernel / np.sum(kernel))


def distort1(img, k=0, dx=0, dy=0):
    height, width = img.shape[:2]

    k = k * 0.00001
    dx = dx * width
    dy = dy * height
    x, y = np.mgrid[0:width:1, 0:height:1]
    x = x.astype(np.float32) - width/2 - dx
    y = y.astype(np.float32) - height/2 - dy
    theta = np.arctan2(y, x)
    d = (x*x + y*y)**0.5
    r = d*(1+k*d*d)
    map_x = r*np.cos(theta) + width/2 + dx
    map_y = r*np.sin(theta) + height/2 + dy

    img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_REFLECT_101)
    return img


def distort2(img, num_steps=10, xsteps=[], ysteps=[]):
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step*xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end-start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step*ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end-start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    img = cv2.remap(img, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    return img

def elastic_transform_fast(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003] (with modifications).
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32([
        center_square + square_size, 
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size])
    pts2 = pts1 + random_state.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, M, shape_size[::-1], 
                           borderMode=cv2.BORDER_REFLECT_101)

    dx = np.float32(gaussian_filter(
        (random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
    dy = np.float32(gaussian_filter(
        (random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)

    return cv2.remap(image, mapx, mapy, 
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


@clipped
def gauss_noise(image, var):
    row, col, ch = image.shape
    mean = var
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma, (row,col,ch))
    gauss = gauss.reshape(row, col, ch)
    gauss = (gauss - np.min(gauss)).astype(np.uint8)
    return image.astype(np.int32) + gauss


@clipped
def random_brightness(img, alpha):
    return alpha * img


@clipped
def random_contrast(img, alpha):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    return alpha * img + gray

