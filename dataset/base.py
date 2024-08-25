from glob import glob
from typing import List
import json
import math
import os
import random

from torch.utils.data import Dataset
from tqdm import tqdm
import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
import scipy.ndimage as ndimage
import torchvision.transforms as T

from dataset import test_transform, DATASET_INFOS, read_image, read_mask
from dataset.transforms import RandomSPNoise, RandomLightness


class CPRDataset(Dataset):
    def __init__(self, dataset_name: str, sub_category: str, resize: int, data_dir: str, scales: List[int], region_sizes: List[int], retrieval_dir: str, foreground_dir: str = None, nAnomaly: int = 0, knn: int = 10) -> None:
        self.dataset_name             = dataset_name
        self.sub_category             = sub_category
        self.is_object                = sub_category in DATASET_INFOS[self.dataset_name][1]
        self.resize                   = resize
        self.data_dir                 = data_dir
        self.foreground_dir           = foreground_dir
        self.retrieval_dir            = retrieval_dir
        self.scales                   = scales
        self.region_sizes             = region_sizes
        self.nAnomaly                 = nAnomaly
        self.knn                      = knn
        self.use_foreground           = foreground_dir is not None
        self.outlier_data             = []
        self.train_infos              = []
        self.test_infos               = []
        self.synthetic_infos          = []
        self.foreground_weights       = {}
        self.outlier_data_cache       = {}
        self._cache                   = {}
        self.root_dir                 = os.path.join('./data', self.dataset_name, sub_category)
        self.foreground_result        = {}

        # load data
        with open(os.path.join(retrieval_dir, sub_category, 'r_result.json'), 'r') as f:
            self.retrieval_result: dict[list] = json.load(f)

        self.train_fns = sorted(glob(os.path.join(self.root_dir, 'train/*/*')))
        self.test_fns = sorted(glob(os.path.join(self.root_dir, 'test/*/*')))
        
        if self.nAnomaly > 0:  # supervised
            _normal_data = []
            _outlier_data = []
            for anomaly_name in sorted(os.listdir(os.path.join(self.root_dir, 'test'))):
                for fn in sorted(os.listdir(os.path.join(self.root_dir, 'test', anomaly_name))):
                    image_fn = os.path.join(self.root_dir, 'test', anomaly_name, fn)
                    if anomaly_name == 'good':
                        _normal_data.append(image_fn)
                    else:
                        _outlier_data.append(image_fn)
            np.random.RandomState(42).shuffle(_outlier_data)  # DRA
            self.test_fns = _outlier_data[self.nAnomaly:] + _normal_data  # delete training samples 
            for image_fn in _outlier_data[:self.nAnomaly]:  # cache
                k = os.path.relpath(image_fn, os.path.join(self.root_dir))
                image_name = os.path.basename(k)[:-4]
                anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
                mask_fn = os.path.join(self.root_dir, 'ground_truth', anomaly_name, image_name + '_mask.png')
                self.outlier_data.append({'image_fn': image_fn, 'mask_fn': mask_fn, 'k': k, 'sub_category': sub_category, 'is_object': self.is_object})
                image = self.read_image(image_fn)
                mask = self.read_mask(mask_fn)
                self.outlier_data_cache[k] = (image, mask)
            
        for image_fn in self.test_fns:
            k = os.path.relpath(image_fn, os.path.join(self.root_dir))
            anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
            image_name = os.path.basename(k)[:-4]
            mask_fn = os.path.join(self.root_dir, 'ground_truth', anomaly_name, image_name + '_mask.png') if anomaly_name != 'good' else None
            self.test_infos.append({'image_fn': image_fn, 'mask_fn': mask_fn, 'k': k, 'sub_category': sub_category, 'is_object': self.is_object})
        
        for image_fn in self.train_fns:
            k = os.path.relpath(image_fn, os.path.join(self.root_dir))
            self.train_infos.append({'image_fn': image_fn, 'mask_fn': None, 'k': k, 'sub_category': sub_category, 'is_object': self.is_object})
            
        assert os.path.exists(os.path.join(self.data_dir, sub_category, 'train.txt'))
        with open(os.path.join(self.data_dir, sub_category, 'train.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                name, k = line.split(' ', 1)
                uid, format = os.path.splitext(name)
                image_fn = os.path.join(self.data_dir, sub_category, f'{name}')
                mask_fn = os.path.join(self.data_dir, sub_category, f'{uid}_mask{format}')
                self.synthetic_infos.append({'image_fn': image_fn, 'mask_fn': mask_fn, 'k': k, 'sub_category': sub_category, 'is_object': self.is_object})
        
        if self.use_foreground and self.is_object:
            for info in tqdm(self.train_infos + self.test_infos + self.outlier_data, desc='load foreground', leave=False):
                k = info['k']
                sub_category = info['sub_category']
                foreground_weight_path = os.path.join(self.foreground_dir, sub_category, os.path.join(os.path.dirname(k), f'f_{os.path.basename(k)[:-4]}.npy'))
                foreground_weight = np.load(foreground_weight_path).astype(float)
                foreground_weight = ndimage.gaussian_filter(foreground_weight, sigma=foreground_weight.shape[0]/25)
                foreground_weight = self.sharpen(foreground_weight)
                foreground_weight = cv.resize(foreground_weight, (self.resize, self.resize))
                self.foreground_weights[k] = foreground_weight
                self.foreground_result[k] = foreground_weight_path
        
        self.aug_transform = T.Compose([
            T.RandomApply([RandomSPNoise(0.97)], .3),
            T.RandomApply([RandomLightness(0.1)], .3),
        ])
        self.transform = test_transform
        
        self.mgrid_points = np.stack(np.mgrid[:self.resize, :self.resize], -1).reshape(-1, 2)

    def __len__(self):
        return len(self.synthetic_infos)
    
    @classmethod
    def sharpen(cls, x):
        x = 1 / (1 + np.exp(5 - 10 * x))
        return x
    
    def read_image(self, fn, cache: bool = False):
        if fn in self._cache:
            return self._cache[fn]
        image = read_image(fn, (self.resize, self.resize))
        if cache:
            self._cache[fn] = image
        return image
    
    def read_mask(self, fn, cache: bool = False):
        if fn in self._cache:
            return self._cache[fn]
        mask = read_mask(fn, (self.resize, self.resize))
        if cache:
            self._cache[fn] = mask
        return mask
    
    def extended_anomaly(self, image, mask):
        image = image.copy()
        mask = mask.copy()
        for i in range(random.randint(1, 5)):
            info = random.choice(self.outlier_data)
            k = info['k']
            outlier_image, outlier_mask = self.outlier_data_cache[k]
            if random.random() < 0.5:
                outlier_image = iaa.Sequential(random.sample([
                    iaa.GammaContrast((0.5,2.0),per_channel=True),
                    iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                    iaa.pillike.EnhanceSharpness(),
                    iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                    iaa.Solarize(0.5, threshold=(32,128)),
                    iaa.Posterize(),
                    iaa.Invert(),
                    iaa.pillike.Autocontrast(),
                    iaa.pillike.Equalize()
                ], 3))(image=outlier_image)
            if random.random() < 0.9:
                y, x = np.where(outlier_mask > 0)
                center = (int(x.mean()), int(y.mean()))
                degress = random.randint(0, 359)
                RM = np.concatenate([cv.getRotationMatrix2D(center, degress, random.uniform(0.5, 1.5)), np.eye(3)[-1:]])
                x, y = np.split(cv.transform(np.stack([x, y], -1)[:, None], RM).astype(int)[:, 0, :2], 2, 1)
                p1 = max(x.min(), 0), max(y.min(), 0)
                p2 = min(x.max(), self.resize), min(y.max(), self.resize)
                TM = np.eye(3)
                TM[0, 2] = random.randint(-p1[0], self.resize - p2[0])
                TM[1, 2] = random.randint(-p1[1], self.resize - p2[1])
                M = TM @ RM
                # warp
                outlier_image = cv.warpPerspective(outlier_image, M, (self.resize, self.resize))
                outlier_mask = cv.warpPerspective(outlier_mask, M, (self.resize, self.resize))
            y, x = np.where(outlier_mask > 0)
            factor = random.uniform(0.5, 1)
            image[outlier_mask > 0] = image[outlier_mask > 0] * (1 - factor) + outlier_image[outlier_mask > 0] * factor
            mask[outlier_mask > 0] = 255
        return image, mask
    
    def __getitem__(self, idx):
        positive_n = anomaly_n = negative_n = 300
        negative_distance_threshold = self.scales[0] * (self.region_sizes[0] // 2) * math.sqrt(2)

        info     = self.synthetic_infos[idx]
        if self.nAnomaly > 0 and random.random() < 0.025: # supervised
            info = random.choice(self.outlier_data)
        image_fn       = info['image_fn']
        mask_fn        = info['mask_fn']
        k              = info['k']
        sub_category   = info['sub_category']
        is_object      = info['is_object']

        image = self.read_image(image_fn)
        mask = self.read_mask(mask_fn)
        
        if self.nAnomaly > 0 and random.random() < 0.2:  # extended_anomaly
            image, mask = self.extended_anomaly(image, mask)
            
        retrieval_k = self.retrieval_result[k][random.randint(0, self.knn-1)]
        retrieval_image_fn = os.path.join('./data', self.dataset_name, sub_category, retrieval_k)
        retrieval_image = self.read_image(retrieval_image_fn, True)

        # aug
        image = self.aug_transform(image)
        retrieval_image = self.aug_transform(retrieval_image)

        _normal_points = np.stack(np.where(mask == 0), -1).reshape(-1, 2)[:, ::-1]
        _retrieval_normal_points = _normal_points.copy()
        
        # positive
        _positive_n = min(len(_normal_points), positive_n * 2)
        _p = np.ones(len(_normal_points))
        if self.use_foreground and is_object:
            weights = np.maximum(self.foreground_weights[k], self.foreground_weights[retrieval_k])
            _p = weights[_normal_points[:, 1], _normal_points[:, 0]]
        _idx = np.random.choice(np.arange(len(_normal_points)), size=_positive_n, replace=False, p=_p / _p.sum()) 
        positive_points = _normal_points[_idx]
        retrieval_positive_points = _retrieval_normal_points[_idx]
        
        # anomaly
        _anomaly_points = np.stack(np.nonzero(mask), -1).reshape(-1, 2)[:, ::-1]  # xy
        _anomaly_n = min(len(_anomaly_points), anomaly_n)
        anomaly_points = _anomaly_points[np.random.permutation(len(_anomaly_points))[:_anomaly_n]]
        retrieval_anomaly_points = anomaly_points.copy()

        # negative
        _negative_n = min(len(_normal_points), negative_n * 2)
        _retrieval_negative_points = retrieval_positive_points.copy()
        _negative_points = self.mgrid_points[np.random.permutation(len(self.mgrid_points))[:len(_retrieval_negative_points)]]
        # leave the points that dist > negative_distance_threshold
        dist = np.linalg.norm(_negative_points - _retrieval_negative_points, axis=1)
        dist_mask = dist > negative_distance_threshold
        negative_points = _negative_points[dist_mask]
        retrieval_negative_points = _retrieval_negative_points[dist_mask]

        # normalize to [-1, 1]
        # positive
        positive_weight = np.zeros(positive_n, dtype=np.float32)
        _positive_n = min(len(positive_points), positive_n)
        positive_points = np.concatenate([positive_points[:_positive_n], np.zeros((positive_n - _positive_n, 2))]).astype(np.float32) / (self.resize - 1) * 2 - 1
        retrieval_positive_points = np.concatenate([retrieval_positive_points[:_positive_n], np.zeros((positive_n - _positive_n, 2))]).astype(np.float32) / (self.resize - 1) * 2 - 1
        positive_weight[:_positive_n] = 1.
        
        # anomaly
        anomaly_weight = np.zeros(anomaly_n, dtype=np.float32)
        _anomaly_n = min(len(anomaly_points), anomaly_n)
        anomaly_points = np.concatenate([anomaly_points[:_anomaly_n], np.zeros((anomaly_n - _anomaly_n, 2))]).astype(np.float32) / (self.resize - 1) * 2 - 1
        retrieval_anomaly_points = np.concatenate([retrieval_anomaly_points[:_anomaly_n], np.zeros((anomaly_n - _anomaly_n, 2))]).astype(np.float32) / (self.resize - 1) * 2 - 1
        anomaly_weight[:_anomaly_n] = 1.

        # negative
        negative_weight = np.zeros(negative_n, dtype=np.float32)
        _negative_n = min(len(negative_points), negative_n)
        negative_points = np.concatenate([negative_points[:_negative_n], np.zeros((negative_n - _negative_n, 2))]).astype(np.float32) / (self.resize - 1) * 2 - 1
        retrieval_negative_points = np.concatenate([retrieval_negative_points[:_negative_n], np.zeros((negative_n - _negative_n, 2))]).astype(np.float32) / (self.resize - 1) * 2 - 1
        negative_weight[:_negative_n] = 1.
        if not is_object:
            negative_weight[:] = 0
        
        image = self.transform(image)
        retrieval_image = self.transform(retrieval_image)
        return image, retrieval_image, positive_points, retrieval_positive_points, positive_weight, \
                negative_points, retrieval_negative_points, negative_weight, \
                anomaly_points, retrieval_anomaly_points, anomaly_weight
                

if __name__ == '__main__':
    # vis debug
    # export PYTHONPATH=.
    from dataset import inverse_test_transform
    import matplotlib.pyplot as plt
    dataset = CPRDataset(
        'mvtec',
        'screw',
        320,
        './log/synthetic_mvtec_640_12000_True_jpg',
        [4, 8],
        [3, 1],
        './log/retrieval_mvtec_DenseNet_features.denseblock1_320',
        './log/foreground_mvtec_DenseNet_features.denseblock1_320',
    )
    
    while True:
        try:
            image, retrieval_image, positive_points, retrieval_positive_points, positive_weight, \
            negative_points, retrieval_negative_points, negative_weight, \
            anomaly_points, retrieval_anomaly_points, anomaly_weight = dataset[random.randint(0, len(dataset)-1)]
            image = inverse_test_transform(image)
            retrieval_image = inverse_test_transform(retrieval_image)
            positive_points = (positive_points[positive_weight > 0] + 1) * (dataset.resize - 1) / 2
            retrieval_positive_points = (retrieval_positive_points[positive_weight > 0] + 1) * (dataset.resize - 1) / 2
            negative_points = (negative_points[negative_weight > 0] + 1) * (dataset.resize - 1) / 2
            retrieval_negative_points = (retrieval_negative_points[negative_weight > 0] + 1) * (dataset.resize - 1) / 2
            anomaly_points = (anomaly_points[anomaly_weight > 0] + 1) * (dataset.resize - 1) / 2
            retrieval_anomaly_points = (retrieval_anomaly_points[anomaly_weight > 0] + 1) * (dataset.resize - 1) / 2
            plt.figure(figsize=(10, 10))
            plt.subplot(3, 2, 1)
            plt.imshow(image)
            plt.subplot(3, 2, 2)
            plt.imshow(retrieval_image)
            plt.subplot(3, 1, 2)
            plt.imshow(np.concatenate([image, retrieval_image], axis=1))
            random_positive_idxs = np.random.choice(np.arange(len(positive_points)), min(len(positive_points), 10), replace=False)
            for idx in random_positive_idxs:
                plt.scatter(positive_points[idx, 0], positive_points[idx, 1], c='g')
                plt.scatter(retrieval_positive_points[idx, 0]+image.shape[1], retrieval_positive_points[idx, 1], c='g')
                plt.plot([positive_points[idx, 0], retrieval_positive_points[idx, 0]+image.shape[1]], [positive_points[idx, 1], retrieval_positive_points[idx, 1]], c='g')
            random_negative_idxs = np.random.choice(np.arange(len(negative_points)), min(len(negative_points), 10), replace=False)
            for idx in random_negative_idxs:
                plt.scatter(negative_points[idx, 0], negative_points[idx, 1], c='r')
                plt.scatter(retrieval_negative_points[idx, 0]+image.shape[1], retrieval_negative_points[idx, 1], c='r')
                plt.plot([negative_points[idx, 0], retrieval_negative_points[idx, 0]+image.shape[1]], [negative_points[idx, 1], retrieval_negative_points[idx, 1]], c='r')
            if len(anomaly_points) > 0:
                random_anomaly_idxs = np.random.choice(np.arange(len(anomaly_points)), min(len(anomaly_points), 10), replace=False)
                for idx in random_anomaly_idxs:
                    plt.scatter(anomaly_points[idx, 0], anomaly_points[idx, 1], c='b')
                    plt.scatter(retrieval_anomaly_points[idx, 0]+image.shape[1], retrieval_anomaly_points[idx, 1], c='b')
                    plt.plot([anomaly_points[idx, 0], retrieval_anomaly_points[idx, 0]+image.shape[1]], [anomaly_points[idx, 1], retrieval_anomaly_points[idx, 1]], c='b')
            plt.subplot(3, 2, 5)
            plt.imshow(image)
            plt.scatter(positive_points[:, 0], positive_points[:, 1], c='g', s=1, alpha=0.5)
            plt.scatter(negative_points[:, 0], negative_points[:, 1], c='r', s=1, alpha=0.5)
            if len(anomaly_points) > 0:
                plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='b', s=1, alpha=0.5)
            plt.subplot(3, 2, 6)
            plt.imshow(retrieval_image)
            plt.scatter(retrieval_positive_points[:, 0], retrieval_positive_points[:, 1], c='g', s=1, alpha=0.5)
            plt.scatter(retrieval_negative_points[:, 0], retrieval_negative_points[:, 1], c='r', s=1, alpha=0.5)
            if len(retrieval_anomaly_points) > 0:
                plt.scatter(retrieval_anomaly_points[:, 0], retrieval_anomaly_points[:, 1], c='b', s=1, alpha=0.5)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.show()
            plt.close()
        except KeyboardInterrupt:
            break
    
