"""
#test
#reference https://github.com/assassint2017/MICCAI-LITS2017/blob/master/README.md
"""

import os
import collections
from time import time
import torch
import numpy as np
import SimpleITK as sitk
from ptimz.model_zoo.resunet3d import resunet_3d

from ptimz.loss.dice_loss import Metric_dice


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
        elif 3 == len(img.shape):
            img = _r(img, target_size)
    else:
        raise ValueError(f'Unknown image type {type(img)}')
    if 'torchtensor' == dt:
        img = torch.Tensor(img)
    return img



def main(para):
    os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

    liver_score = collections.OrderedDict()
    liver_score['dice'] = []

    net=resunet_3d(pretrained='ctliver')
    device = torch.device('cuda:0')
    print(device)
    if torch.cuda.device_count() > 1:
        print('let us use',torch.cuda.device_count(),'gpus')
        net = torch.nn.DataParallel(net)
    net.to(device)

    net.eval()

    file_name = []

    output_path = para.output_path
    test_ct_path = para.test_ct_path
    test_seg_path = para.test_seg_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    k = 0
    start = time()
    for file_index, file in enumerate(os.listdir(test_ct_path)):
        k += 1
        file_name.append(file)
        dice = valid(test_ct_path,test_seg_path,output_path,file,net,device,para)
        liver_score['dice'].append(dice)
    dice_mean = np.mean(liver_score['dice'])
    print('average dice {:.3f}'.format(dice_mean))
    end = time()
    print('time:{:.3f} sec/round'.format((end - start) / k))




def valid(test_ct_path,test_seg_path,output_path,file,net,device,para):
    net.eval()

    ct_path = os.path.join(test_ct_path, file)
    ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200

    # sliding window to predict
    start_slice = 0
    end_slice = start_slice + para.size - 1
    count = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)
    probability_map = np.zeros((ct_array.shape[0], 512, 512), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:
            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1])
            ct_tensor = ct_tensor.to(device)
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs
            start_slice += para.stride
            end_slice = start_slice + para.size - 1
        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1
            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1])
            ct_tensor = ct_tensor.to(device)
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())
            del outputs

        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (para.threshold * count)] = 1

    seg_path = os.path.join(test_seg_path, file.replace('volume', 'segmentation'))
    seg = sitk.ReadImage(os.path.join(test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    pred_path = os.path.join(output_path, file.replace('volume', 'pred'))  # ---

    if pred_seg.shape[0] != seg_array.shape[0]:
        pred_seg = sitk_resize(pred_seg, seg_array.shape, axis=-1)
        pred_seg = pred_seg[0:seg_array.shape[0]]


    liver_metric = Metric_dice(seg_array, pred_seg)

    dice = liver_metric.get_dice_coefficient()[0]

    print('ct path {} seg path {} dice {:.3f}'.format(ct_path, seg_path, dice))
    # save output
    new_ct = sitk.GetImageFromArray(pred_seg)
    new_ct.SetDirection(seg.GetDirection())
    new_ct.SetOrigin(seg.GetOrigin())
    new_ct.SetSpacing((seg.GetSpacing()[0], seg.GetSpacing()[1], seg.GetSpacing()[2]))
    sitk.WriteImage(new_ct, pred_path)
    print('write to {}'.format(pred_path))
    return dice




if __name__ == '__main__':
    import argparse

    parse_in = argparse.ArgumentParser()
    print('work folder {}'.format(os.getcwd()))

    parse_in.add_argument('--test_ct_path', type=str, help='path to test ct folder',
                          default='/home/dl/Workspace/jessie/lits17validpre1/ct')
    parse_in.add_argument('--test_seg_path', type=str, help='path to test mask folder',
                          default='/home/dl/Workspace/jessie/lits17validpre1/seg')
    parse_in.add_argument('--output_path', type=str, help='path to test mask folder',
                          default='/home/dl/Workspace/jessie/lits17validpre1/output')
    parse_in.add_argument('--module_path', type=str, help='pretrained model path',
                          default='/home/dl/Workspace/jessie//MICCAI-LITS2017-master/models_liver/net990-0.013-0.018.pth')

    parse_in.add_argument('--gpu', type=str, help='gpu', default='0')

    parse_in.add_argument('--stride', type=int, help='sliding stride',default=12)
    parse_in.add_argument('--size', type=int, help='depth', default=48)
    parse_in.add_argument('--threshold', type=float, help='cutoff thresh', default=0.5)

    opt = parse_in.parse_args()
    main(opt)
