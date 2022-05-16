import sys
sys.path.append(r'D:\Data_analysis\phi-pytroch-image-model-zoo-v0.0.1\phi-pytroch-image-model-zoo\ptimz')
from model_zoo import registry, factory
import numpy as np
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import pydicom



def sitk_read(patient_path):
    reader = sitk.ImageSeriesReader()
    slice_names = reader.GetGDCMSeriesFileNames(patient_path)
    # print(slice_names)
    reader.SetFileNames(slice_names)
    image = reader.Execute()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    # keys = image.GetMetaDataKeys()
    image_array = sitk.GetArrayFromImage(image)
    return image_array, origin, spacing


def read_dcm(patient_path):
    slices_path = os.listdir(patient_path)
    slices_npy = []
    img_posi_list = []
    for slice in slices_path:
        if slice.startswith("I"):
            slice_dicom = pydicom.read_file(os.path.join(patient_path, slice))
            # print('slice_dicom', slice_dicom)
            slice_npy = slice_dicom.pixel_array
            slices_npy.append(slice_npy)
            img_posi_list.append(slice_dicom.ImagePositionPatient[-1])
    slices_npy_sort = sort_posi(slices_npy, img_posi_list)
    slices_npy_sort = np.array(slices_npy_sort)
    # plt.imshow(slices_npy_sort[:, :, 250], cmap='gray')
    # plt.show()
    return slices_npy_sort


def sort_posi(img_npy_list, img_posi_list):
    sorted_id = sorted(range(len(img_posi_list)), key=lambda k: img_posi_list[k], reverse=True)
    img_npy_list_sort = []
    for q in range(len(sorted_id)):
        img_npy_list_sort.append(img_npy_list[sorted_id[q]])
    img_npy_list_sort = np.array(img_npy_list_sort)
    # print('train_patient_path_sort', train_patient_path_sort)
    # print('img_npy_list_sort.shape', img_npy_list_sort.shape)
    return img_npy_list_sort


def hist_plot_crop_norm(data):
    # plt.imshow(data[100, :, :], cmap='gray')
    # plt.show()
    data_reshape = data.reshape(-1)
    fig, axes = plt.subplots()
    axes.hist(data_reshape, range=(0, 4000), facecolor='g')
    axes.set_ylim([0, 60000000])
    # plt.savefig(path_before)
    # plt.close()

    # crop
    threshold = 1500
    # data[data < threshold] = 0
    # plt.imshow(data[100, :, :], cmap='gray')
    # plt.show()
    data[data > threshold] = threshold
    # plt.imshow(data[100, :, :], cmap='gray')
    # plt.show()
    data_reshape = data.reshape(-1)
    # fig, axes = plt.subplots()
    # axes.hist(data_reshape, range=(0, 1500), facecolor='r')
    # axes.set_ylim([0, 60000000])
    # plt.savefig(path_after)
    # plt.close()

    # normalization
    data = data/threshold
    # print('mean-std', np.mean(data), np.std(data))
    # print('min-max', np.min(data), np.max(data))
    return data


def pre_processing(path):
    # print(patient)
    # patient_name = patient.split(('\\'))[-1]
    # print('patient_name', patient_name)
    # _, origin, spacing = sitk_read(patient)
    patient_npy = read_dcm(path)
    # print(patient_npy.shape)
    patient_npy_crop_norm = hist_plot_crop_norm(patient_npy)

    return patient_npy_crop_norm




print(registry.list_models())
print(registry.list_pretrained_names())

img_path = r'D:\Data_analysis\phi-pytroch-image-model-zoo-v0.0.1\data\cyclegan\cyclegan_demo_data\patient1'
save_path = r'D:\Data_analysis\phi-pytroch-image-model-zoo-v0.0.1\data\cyclegan\output'
my_model = factory.create_model("cyclegan2d")
slices = pre_processing(img_path)
for idx in range(slices.shape[0]):
    my_input = slices[idx]

    with torch.no_grad():
        my_input = np.expand_dims(my_input, axis=0)
        my_input = np.expand_dims(my_input, axis=0)
        # print('my_input', my_input.shape)
        my_input = torch.from_numpy(my_input).type(torch.FloatTensor)
        my_output = my_model(my_input)
        plt.imshow(np.concatenate((my_input[0, 0, :, :], my_output[0, 0, :, :]), axis=1), cmap='gray')
        plt.show()
        my_output = sitk.GetImageFromArray(my_output[0, 0, :, :])
        sitk.WriteImage(my_output, os.path.join(save_path, str(idx) + '.nii.gz'))
