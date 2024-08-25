from glob import glob
from itertools import cycle
from pprint import pformat
import argparse
import concurrent.futures
import os
import random
import sys
sys.path.append('./')

from einops import rearrange
from loguru import logger
from tqdm import tqdm
import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
import torch

from dataset import DATASET_INFOS, read_image
from libs.perlin import rand_perlin_2d_np
from utils import fix_seeds, save_dependencies_files


# hyperparameters
min_area_ratio         = 0.001  # filter out anomaly regions with an area less than 0.001
transparency_range     = [0.5, 1.0]
anomaly_ratio          = 0.9
perlin_noise_threshold = 0.5
perlin_scale           = 6
min_perlin_scale       = 0
structure_grid_size    = 16
save_format            = 'jpg'

def generate_perlin_noise_mask(resize) -> np.ndarray:
    # define perlin noise scale
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    # generate perlin noise
    perlin_noise = rand_perlin_2d_np((resize, resize), (perlin_scalex, perlin_scaley))
    
    # apply affine transform
    rot = iaa.Affine(rotate=(-90, 90))
    perlin_noise = rot(image=perlin_noise)
    
    # make a mask by applying threshold
    mask_noise = np.where(
        perlin_noise > perlin_noise_threshold, 
        np.ones_like(perlin_noise), 
        np.zeros_like(perlin_noise)
    )
    
    return mask_noise

def structure_source_img(img: np.ndarray) -> np.ndarray:
    resize = img.shape[0]
    structure_source_img = iaa.Sequential(random.sample([
        iaa.GammaContrast((0.5,2.0),per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
        iaa.pillike.EnhanceSharpness(),
        iaa.AddToHueAndSaturation((-50,50),per_channel=True),
        iaa.Solarize(0.5, threshold=(32,128)),
        iaa.Posterize(),
        iaa.Invert(),
        iaa.pillike.Autocontrast(),
        iaa.pillike.Equalize(),
        iaa.Affine(rotate=(-45, 45))
    ], 3))(image=img)
    
    assert resize % structure_grid_size == 0, 'structure should be devided by grid size accurately'
    grid_w = resize // structure_grid_size
    grid_h = resize // structure_grid_size
    
    structure_source_img = rearrange(
        tensor  = structure_source_img, 
        pattern = '(h gh) (w gw) c -> (h w) gw gh c',
        gw      = grid_w, 
        gh      = grid_h
    )
    disordered_idx = np.arange(structure_source_img.shape[0])
    np.random.shuffle(disordered_idx)

    structure_source_img = rearrange(
        tensor  = structure_source_img[disordered_idx], 
        pattern = '(h w) gw gh c -> (h gh) (w gw) c',
        h       = structure_grid_size,
        w       = structure_grid_size
    ).astype(np.float32)
    return structure_source_img

def generate_synthetic_anomaly_img(img, is_object, foreground_weight):
    resize = img.shape[0]
    while True:
        anomaly_img_mask = generate_perlin_noise_mask(resize)
        cn, cc_labels, _, _ = cv.connectedComponentsWithStats((anomaly_img_mask * 255).astype(np.uint8), connectivity=8)
        for i in range(1, cn):
            cur_anomaly_mask = cc_labels == i
            y, x = np.where(cur_anomaly_mask)
            # filter out small area, on foreground
            if len(y) < (min_area_ratio * resize * resize) or (foreground_weight[y, x] > 0.5).sum() == 0:
                anomaly_img_mask[cur_anomaly_mask] = 0
        if np.any(anomaly_img_mask > 0):
            break
    mask_expanded = np.expand_dims(anomaly_img_mask, axis=2)
    # step 2. generate texture or structure anomaly
    ## anomaly source
    if np.random.uniform() < 0.5 or not is_object:
        anomaly_source_img = read_image(random.choice(sorted(glob(os.path.join('./data/dtd/images', '*/*')))), (resize, resize)).astype(np.float32)
    else:
        anomaly_source_img = structure_source_img(img)
    ## mask anomaly parts
    factor = np.random.uniform(*transparency_range, size=1)[0]
    anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
    # step 3. blending image and anomaly source
    anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
    return anomaly_source_img.astype(np.uint8), (anomaly_img_mask * 255).astype(np.uint8)

def generate_one(uid, save_path, info, resize, is_object, foreground_fn: str = None):
    fix_seeds(uid)  # reproduce
    image_fn, k = info
    assert os.path.join(image_fn), f'{image_fn} not exists'
    img = read_image(image_fn, (resize, resize))
    anomaly_img_mask = np.zeros((resize, resize))
    anomaly_source_img = img
    if foreground_fn is not None and is_object:
        foreground_weight = cv.resize(np.load(foreground_fn), (resize, resize))
    else:
        foreground_weight = np.ones((resize, resize))
    if random.random() < anomaly_ratio:
        anomaly_source_img, anomaly_img_mask = generate_synthetic_anomaly_img(img, is_object, foreground_weight)
    cv.imwrite(os.path.join(save_path, f'{uid}.{save_format}'), cv.cvtColor(anomaly_source_img, cv.COLOR_RGB2BGR))
    cv.imwrite(os.path.join(save_path, f'{uid}_mask.{save_format}'), anomaly_img_mask)
    return uid, k

def generate(num_workers, save_path, dataset_name, resize, num, foreground_dir: str = None):
    logger.info(f'gen_simulated_anomaly')
    logger.info(f'save to {save_path}')
    logger.info(f'params: {num_workers} {dataset_name} {resize} {num} {foreground_dir}')
    data_root = os.path.join('./data', dataset_name)
    dataset_info = DATASET_INFOS[dataset_name]
    for sub_category_id, sub_category in enumerate(dataset_info[0]):  # all
        is_object = sub_category in dataset_info[1]
        logger.info(f'processing {sub_category}')
        sub_category_save_path = os.path.join(save_path, sub_category)
        os.makedirs(sub_category_save_path, exist_ok=True)
        train_infos = [(fn, os.path.relpath(fn, os.path.join(data_root, sub_category))) for fn in sorted(glob(os.path.join(data_root, sub_category, 'train/*/*')))]  # (fn, k)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor, open(os.path.join(sub_category_save_path, 'train.txt'), 'w') as f:
            params = [
                range(sub_category_id * num, (sub_category_id + 1) * num),
               (sub_category_save_path for _ in range(num)),
               (info for info, _ in zip(cycle(train_infos), range(num))),
               (resize for _ in range(num)),
               (is_object for _ in range(num)),
               (foreground_dir and os.path.join(foreground_dir, sub_category, os.path.dirname(info[1]), 'f_' + os.path.basename(info[1]).split('.')[0] + '.npy') for info, _ in zip(cycle(train_infos), range(num))),
            ]
            for (uid, k) in tqdm(executor.map(generate_one, *params), total=num, desc=f'{sub_category} generate', leave=False):
                f.write(f'{uid}.{save_format} {k}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("--num-workers", type=int, default=min(32, os.cpu_count()), help="num workers")
    parser.add_argument("-lp", "--log-path", type=str, default=None, help="log path")
    # data
    parser.add_argument("--dataset-name", type=str, default="mvtec", choices=list(DATASET_INFOS.keys()), help="dataset name")
    parser.add_argument("--resize", type=int, default=640, help="image resize")
    parser.add_argument("--num", type=int, default=12000, help="num of synthetic samples")
    parser.add_argument("-fd", "--foreground-dir", type=str, default=None, help="foreground dir")
    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = f'log/synthetic_{args.dataset_name}_{args.resize}_{args.num}_{args.foreground_dir is not None}_{save_format}'
    logger.add(os.path.join(args.log_path, 'runtime.log'))
    logger.info('args: \n' + pformat(vars(args)))
    save_dependencies_files(os.path.join(args.log_path, 'src'))
    generate(args.num_workers, args.log_path, args.dataset_name, args.resize, args.num, args.foreground_dir)