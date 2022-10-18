import SimpleITK as sitk
import numpy as np
import torch

__all__ = ['sitk_respacing', 'image_normalize', 'image_standardize', 'sitk_resize', 'itkimage_resize']


def sitk_respacing(image: sitk.Image, spacing, interpolator=sitk.sitkLinear):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(spacing)
    ori_size = image.GetSize()
    ori_spacing = image.GetSpacing()
    new_size = ori_size * (np.asarray(ori_spacing) / np.asarray(spacing))
    new_size = np.ceil(new_size).astype(np.int)
    resample.SetSize(new_size.tolist())
    resample.Execute(image)
    return resample.Execute(image)


def image_normalize(im, rang=1.):
    if isinstance(im, np.ndarray):
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
    elif isinstance(im, torch.Tensor):
        im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))
    else:
        raise ValueError(f"image_normalize Unknown type {type(im)}")
    return im * rang


def image_standardize(im, mean, std):
    im = im.astype(float)
    im = (im - mean) / std
    return im


def window_convert(im, ww=None, wl=None, upper=None, lower=None):
    if ww is not None and wl is not None:
        upper = wl + ww / 2
        lower = wl - ww / 2
    if upper is None or lower is None:
        raise ValueError(f"Got NULL value upper={upper} lower={lower}")
    # np.clip(im, lower, upper)
    if lower is not None:
        im[im < lower] = lower
    if upper is not None:
        im[im > upper] = upper
    return im


def itkimage_resize(img, target_size, interp=sitk.sitkLinear):
    d, h, w = target_size
    resampler = sitk.ResampleImageFilter()
    origin_size = img.GetSize()
    origin_spacing = img.GetSpacing()
    factor = origin_size / np.array([w, h, d])
    new_spacing = origin_spacing * factor
    resampler.SetReferenceImage(img)
    resampler.SetSize([w, h, d])
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interp)
    return resampler.Execute(img)


def sitk_resize(img, target_size, axis=-1):
    def _r(img, target_size, interp=sitk.sitkLinear):
        d, h, w = target_size
        img = sitk.GetImageFromArray(img)
        resampler = sitk.ResampleImageFilter()
        origin_size = img.GetSize()
        origin_spacing = img.GetSpacing()
        factor = origin_size / np.array([w, h, d])
        new_spacing = origin_spacing * factor
        resampler.SetReferenceImage(img)
        resampler.SetSize([w, h, d])
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(interp)
        img = sitk.GetArrayFromImage(resampler.Execute(img))
        return img

    dt = None
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        dt = 'torchtensor'
    if isinstance(img, np.ndarray):
        assert 3 == len(target_size), f"target size should be (d, h, w), got {target_size}"
        if 4 == len(img.shape):
            # dhwc
            arrs = []
            for cid in range(img.shape[axis]):
                if axis == -1:
                    arrs.append(_r(img[:, :, :, cid], target_size, sitk.sitkLinear))
                elif axis == 0:
                    arrs.append(_r(img[cid], target_size, sitk.sitkNearestNeighbor))
                else:
                    raise ValueError(f'axis {axis} is not supported')
            img = np.stack(arrs, axis=axis)
        elif 3 == len(target_size):
            img = _r(img, target_size)
    else:
        raise ValueError(f'Unknown image type {type(img)}')
    if 'torchtensor' == dt:
        img = torch.Tensor(img)
    return img
