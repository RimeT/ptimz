# Quality assessment based on CMR images
# Details of image
# spacing=(2.0, 2.0),
# slice_thickness=8,
# slice_gap=4,
# mild motion=1 intermediate motion=2 severe motion=3

import argparse
import logging
import os
from time import time

import numpy as np
import pandas as pd
import torch
from monai.transforms import *
from torchvision import transforms as ttf

import ptimz
from ptimz.data import load_image, image_normalize


def get_logger():
    logger = logging.getLogger('cmr qa prediction')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cmr quality assessment')
    parser.add_argument('input', type=str,
                        help='Root dir of scans. All the files must have suffix and file type could be nifty, nrrd, dcms in a directory.')
    parser.add_argument('outcsv', type=str, help='csv file name with no suffix')
    parser.add_argument('weights_file', type=str, help='model weights')

    parser.add_argument('--gpuid', type=int, default=-1, help='GPU id, -1 will run on CPU.')
    parser.add_argument('--topk', type=int, default=-1,
                        help='This will select topk high score slices to make the final prediction. set -1 to disable topk.')

    opt = parser.parse_args()

    src_root = opt.input
    out_csv = opt.outcsv + '_binary.csv'
    scoreout_csv = opt.outcsv + '_scores.csv'

    weights_file = opt.weights_file

    topk = opt.topk

    device_id = opt.gpuid
    avail_gpus = torch.cuda.device_count()
    label_map = [1, 2, 3]

    logger = get_logger()

    # walk input scans
    image_paths = []
    for ds, _, fs in os.walk(src_root):
        for f in fs:
            image_paths.append(os.path.join(ds, f))
    logger.info(f"Scans len = {len(image_paths)}")

    model = ptimz.create_model("efficientnetb0_2d", pretrained='cmrqa', in_chans=1, num_classes=3)
    model.eval()

    if device_id < 0 or device_id > avail_gpus:
        pass
    else:
        model.cuda(device_id)

    logger.info('Model ready.')

    v_fn = Compose([
        ttf.ToTensor(),
        ResizeWithPadOrCrop((512, 512)),
    ])

    predictions = []
    scores_record = []
    df = []

    t1 = time()
    sample_t1 = time()
    for fid, f in enumerate(sorted(image_paths)):
        pid = os.path.basename(f)
        if pid.startswith('.'):
            continue
        image = load_image(f)
        # pixel value normalization by 3d
        image = image_normalize(image)

        slice_preds = []
        for sli in image:
            if device_id < 0 or device_id > avail_gpus:
                sli = v_fn(sli).unsqueeze(0)
            else:
                sli = v_fn(sli).cuda(device_id).unsqueeze(0)
            with torch.no_grad():
                output = model(sli)
            output = torch.softmax(output, 1).squeeze(0)
            slice_preds.append(output.cpu().numpy())
        slice_preds = np.stack(slice_preds).transpose((1, 0))
        score_preds = []
        for cate_id, cate_preds in enumerate(slice_preds):
            cate_preds = cate_preds[np.argsort(cate_preds)[::-1]]
            if topk > 0:
                score_preds.append(np.mean(cate_preds[:topk]))
            else:
                score_preds.append(np.mean(cate_preds))

        # record preds
        predictions.append(np.argmax(score_preds))
        scores_record.append(np.array(score_preds))

        df.append({"Image": pid, "Label": label_map[np.argmax(score_preds)]})

        if fid % len(image_paths) // 25 == 0:
            logger.info(f'Progress: {fid / len(image_paths) * 100: 02.02f}% {time() - sample_t1:.02f} samples/s')

        sample_t1 = time()

    logger.info(f'Finish. {len(image_paths)} samples in {time() - t1:.02f} s')

    # write to file
    df = pd.DataFrame(df)
    df.to_csv(out_csv, index=None)
    logger.info(f'Binary prediction write to {out_csv}')
    scores_series = np.stack(scores_record).T
    score_df = {'Image': df['Image']}
    for idx, i in enumerate(label_map):
        score_df[i] = pd.Series(scores_series[idx])
    score_df = pd.DataFrame(score_df)
    score_df.to_csv(scoreout_csv, index=None)
    logger.info(f'Scores write to {out_csv}')
