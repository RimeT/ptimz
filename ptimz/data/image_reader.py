import traceback

import SimpleITK as sitk
import cv2
import nrrd
import numpy as np
import os
import pydicom

IMTYPE_CV = 'cv'
IMTYPE_DCM = 'dicom'
IMTYPE_NIFTY = 'nifty'
IMTYPE_MHD = 'mhd'
IMTYPE_NRRD = 'nrrd'
IMTYPE_GRAYCV = 'graycv'
IMTYPE_NUMPY = 'numpy'

IMAGE_EXTENSIONS = {
    'cv': ['.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF'],
    'dicom': ['.DCM'],
    'nifty': ['.NII', '.NII.GZ'],
    'mhd': ['.MHD'],
    'nrrd': ['.NRRD'],
    'numpy': ['.NPY', '.NPZ']
}


def get_backend_of_image(image_path):
    if os.path.isdir(image_path):
        files_in_path = [fn for fn in os.listdir(image_path) if not fn.startswith('.')]
    else:
        files_in_path = [image_path]

    for im_fmt in IMAGE_EXTENSIONS:
        ext_list = IMAGE_EXTENSIONS[im_fmt]
        match = True
        for fn in files_in_path:
            if not any(fn.upper().endswith(x) for x in ext_list):
                match = False
        if match:
            return im_fmt
    raise TypeError(f'Image type only supports {IMAGE_EXTENSIONS.values()}.')


def dcm_slope(dicom_img: pydicom.dataset.Dataset):
    slope = 1
    intercept = 0
    image = dicom_img.pixel_array.copy()
    try:
        intercept = dicom_img.RescaleIntercept
        slope = dicom_img.RescaleSlope
    except Exception:
        pass
        # traceback.print_exc()
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image = image + np.int16(intercept)
    return image


def _single_image_reader(path, lib):
    if IMTYPE_CV == lib:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    elif IMTYPE_GRAYCV == lib:
        return cv2.imread(path, flags=0)
    elif IMTYPE_DCM == lib:
        try:
            return dcm_slope(pydicom.dcmread(path))
        # TODO: change exception fields
        except Exception:
            traceback.print_exc()
            return sitk.GetArrayFromImage(sitk.ReadImage(path))
    elif IMTYPE_NRRD == lib:
        return nrrd.read(path)
    elif IMTYPE_NIFTY == lib:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    elif IMTYPE_MHD == lib:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    elif IMTYPE_NUMPY == lib:
        return np.load(path)
    else:
        raise TypeError(f'Got {lib}. Image type only supports {IMAGE_EXTENSIONS.values()}.')


def load_image(path, lib=None):
    """
    3d Dicom will return DHWC
    :param path: image path
    :param lib: library use to load image [cv dicom nrrd nifty]
    :return: numpy array
    """
    if lib is None:
        backend = get_backend_of_image(path)
    else:
        backend = lib
    if os.path.isdir(path):
        if IMTYPE_DCM == backend:
            # return in DHW
            dcm_reader = sitk.ImageSeriesReader()
            dcm_reader.SetFileNames(dcm_reader.GetGDCMSeriesFileNames(path))
            image = dcm_reader.Execute()
            return sitk.GetArrayFromImage(image)
        else:
            # TODO: sort file logic
            image = []
            for im_path in sorted(os.listdir(path)):
                image.append(np.expand_dims(_single_image_reader(os.path.join(path, im_path), backend), axis=0))
            return np.asarray(image)
    else:
        return _single_image_reader(path, backend)


def paths2numpy(paths: list, lib=None, fill_na: bool = False) -> np.ndarray:
    """

    :param paths: list of image paths
    :param lib: specify image reader library
    :param fill_na: if image path is invalid, fill it with zero mask
    :return: 3D image in format DHW
    """
    # TODO verify this function
    images = []
    width = None
    height = None
    for p in paths:
        if os.path.isfile(p):
            im = image_reader(p, lib)
            if width is not None and height is not None:
                # im should be 2d
                height, width = im.shape[:2]
            images.append(im)
        elif fill_na:
            images.append(None)
    for i in range(len(images)):
        if images[i] is not None:
            images[i] = cv2.resize(images[i], (width, height))
        else:
            images[i] = np.zeros((height, width))
    images = np.stack(images, axis=0)
    return images
