# -*- coding: utf-8 -*-
# @Time    : 2017/10/10 22:30
# @Author  : A_star
import cv2
import numpy as np
from PIL import Image

def rotation(img, rotation_min=-10, rotation_max=10):
    '''
    Rotate the image at an angle
    return image and angle
    '''
    rows, cols, channel = img.shape
    ro_degr = np.random.randint(rotation_min, rotation_max)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ro_degr, 1)
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    return dst, ro_degr


def scale(img,  scale_min=80, scale_max=120):
    '''
    Scales the image with a certain range
    :param img: ndarray
    :param scale_min: int
    :param scale_max: int
    :return: ndarray
    '''
    rows, cols, channel = img.shape
    scale_factor = np.random.randint(scale_min, scale_max) * 0.01
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale_factor)
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT)
    return dst, scale_factor


def transform(img, rect, dst):
    '''
    A perspective transformation of an image
    :param img: image
    :param rect: source four points   eg:np.array([(1,2),(3,4),(5,6),(7,8)], dtype="int16")
    :param dst: destination  four points
    :return:ndarray
    '''
    size_x, size_y, channel = img.shape
    M = cv2.getPerspectiveTransform(rect, dst)  # rect  dst
    warped = cv2.warpPerspective(img, M, (size_x, size_y))
    return warped


def add_background_pil(image, back_image, xx=0, yy=0):
    '''
    add background
    :param image:
    :param back_image:
    :param xx:int
    :param yy:int
    :return:
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)  # change colour
    image = Image.fromarray(image)
    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)  # change colour
    back_image = Image.fromarray(back_image)
    back_image.paste(image, (xx,yy), image)
    return back_image





