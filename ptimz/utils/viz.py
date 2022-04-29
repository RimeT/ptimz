import cv2
import numpy as np
from matplotlib import pyplot as plt
from ptimz.data.transforms import image_normalize


def plot_image(img, title=None):
    img = np.asarray(img)
    img = image_normalize(img, 255).astype('uint8')
    if 2 == len(img.shape):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if title:
        plt.title(title)
    plt.imshow(img)


def plot_semseg(img, mask=None, pos=0.5, alpha=0.5):
    if 3 == len(img.shape):
        sli = round(img.shape[0] * pos)
        print(f'display slice {sli}')
        imgsli = img[sli]
        if mask is not None:
            if len(mask) == 1:
                disp = mask_overlay(image_normalize(imgsli, 255).astype('uint8'), mask[:, sli], alpha=alpha)
            else:
                disp = mask_overlay(image_normalize(imgsli, 255).astype('uint8'), mask[1:, sli], alpha=alpha)
            plt.imshow(disp)
        else:
            plot_image(imgsli)


def mask_overlay(img, masks, alpha: float = 0.5):
    """
    Visualize segmentation mask
    :param img: pixel value 0 - 255
    :param masks: binary mask
    :param alpha:
    :return:
    """
    if 2 == len(img.shape):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for mask in masks:
        color = np.random.random(3) * 255
        mask = np.repeat((mask > 0)[:, :, np.newaxis], repeats=3, axis=2)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)
    return img.astype('uint8')