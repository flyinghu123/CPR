import random

import cv2 as cv
import numpy as np
import torch.nn as nn


class RandomSPNoise(nn.Module):
    def __init__(self, snr: float = 0.95):
        super().__init__()
        self.snr = snr
    
    def forward(self, img):
        img = img.copy()
        h, w, c = img.shape
        cur_snr = random.uniform(self.snr, 1.0)
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[cur_snr, (1 - cur_snr) / 2., (1 - cur_snr) / 2.])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 1] = 255
        img[mask == 2] = 0
        return img


class RandomLightness(nn.Module):
    def __init__(self, ratio: float = 0.1):
        super().__init__()
        self.ratio = ratio
    
    def forward(self, img):
        hls_img = cv.cvtColor(img, cv.COLOR_RGB2HLS).astype(float)
        hls_img[:, :, 1] = (1.0 + random.uniform(-self.ratio, self.ratio)) * hls_img[:, :, 1]
        hls_img = np.clip(hls_img, 0, 255).astype(np.uint8)
        return cv.cvtColor(hls_img, cv.COLOR_HLS2RGB)
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = cv.imread('./data/mvtec/bottle/train/good/001.png')
    image = RandomSPNoise()(image)
    plt.imshow(image[..., ::-1])
    plt.show()
    
    
    
    
