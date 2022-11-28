# Fetal brain white matter segmentation
# Details of image
# in-plane resolution of 0.5mm x 0.5mm, and a slice thickness of 3 to 5 mm.
# The sequence parameters were the following: TR: 2000-3500ms; TE: 120ms (minimum);
# flip angle: 90°; sampling percentages 55%.
# The following parameters were adjusted depending on the gestational age and size of the fetus: FOV: 200-240 mm; Image Matrix: 1.5T: 256x224;
# 3T: 320x224. The imaging plane was oriented relative to the fetal brain and axial, coronal, and sagittal images were acquired
# An SR reconstruction algorithm was then applied to each subject’s stack of images and brain masks, creating a 3D SR volume of brain morphology (sub-001 to 040: [2];
# sub-041 to sub-080: [1])
# with an isotropic resolution of 0.5mm^3. Each image was histogram-matched using Slicer, and zero-padded to be 256x256x256 voxels.
# mask 0: background 1: white matter

import argparse
import logging
import os
from time import time

import SimpleITK as sitk
import numpy as np
import torch

import ptimz


def get_logger():
    logger = logging.getLogger('feta wm prediction')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def load_itk(path):
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path))
        image = reader.Execute()
    elif path.endswith('nii.gz') or path.endswith('.nii'):
        image = sitk.ReadImage(path)
    else:
        raise TypeError(f"Only support dicom and nifti files. but got {os.path.basename(path)}")
    return image


def out_name(in_path):
    base = os.path.basename(in_path)
    if os.path.isdir(in_path):
        return base
    else:
        if base.endswith('.nii.gz'):
            return base
        elif base.endswith('.nii'):
            return base + '.gz'
        else:
            raise TypeError(f"Only support dicom and nifti files. but got {base}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cmr quality assessment')
    parser.add_argument('input', type=str,
                        help='Root dir of scans. All the files must have suffix and file type could be nifty, nrrd, dcms in a directory.')
    parser.add_argument('output', type=str, help='outputdir')
    parser.add_argument('weights_file', type=str, help='model weights')

    parser.add_argument('--gpuid', type=int, default=-1, help='GPU id, -1 will run on CPU.')

    opt = parser.parse_args()

    src_root = opt.input
    out_root = opt.output
    weights_file = opt.weights_file
    device_id = opt.gpuid

    logger = get_logger()

    avail_gpus = torch.cuda.device_count()

    if device_id > 0 and torch.cuda.is_available() and device_id < avail_gpus:
        avail_mem, total_mem = torch.cuda.mem_get_info(device_id)
        avail_mem = avail_mem / 1024 / 1024
        if avail_mem < 4000:
            logger.info(f'Available GPU mem should be greater than 4000MB, Now {avail_mem}MB')
            device_id = -1
    else:
        logger.info("CUDA not available.")

    image_paths = []

    for f in os.listdir(src_root):
        fp = os.path.join(src_root, f)
        if os.path.exists(fp):
            image_paths.append(fp)

    logger.info(f"Scans len = {len(image_paths)}")

    model = ptimz.create_model("litehrnet_seg3d", pretrained=None, in_chans=1, num_classes=2)
    checkpoint = torch.load(weights_file, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    logger.debug(f'load state dict from {weights_file}')

    if device_id < 0 or device_id > avail_gpus:
        pass
    else:
        model.cuda(device_id)

    logger.info('Model ready.')

    from monai import transforms as mnf

    fn = mnf.Compose([
        # resize image to 224x224x224
        mnf.AddChannel(),
        mnf.ResizeWithPadOrCrop((224, 224, 224)),
        # mnf.Resize((224, 224, 224)),
        # rescale the intensity
        mnf.ScaleIntensity()
    ])

    # run inference
    t1 = time()
    sample_t1 = time()
    for fid, f in enumerate(sorted(image_paths)):
        imageitk = load_itk(f)
        input_tensor = sitk.GetArrayFromImage(imageitk)
        origin_shape = input_tensor.shape
        input_tensor = torch.as_tensor(fn(input_tensor))
        if device_id < 0 or device_id > avail_gpus:
            pass
        else:
            input_tensor = input_tensor.cuda(device_id)

        # add batch channel
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        output = torch.argmax(output, 1)[0].cpu().numpy()

        # resize the mask back
        temp_fn = mnf.ResizeWithPadOrCrop(origin_shape)
        mask = temp_fn(np.expand_dims(output, 0))[0]

        maskitk = sitk.GetImageFromArray(mask)
        maskitk.CopyInformation(imageitk)
        sitk.WriteImage(maskitk, os.path.join(out_root, out_name(f)))

        if fid % len(image_paths) // 25 == 0:
            logger.info(f'Progress: {fid / len(image_paths) * 100: 02.02f}% {time() - sample_t1:.02f} samples/s')

        sample_t1 = time()

    logger.info(f'Finish. {len(image_paths)} samples in {time() - t1:.02f} s')
