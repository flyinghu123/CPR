from torch.utils.data.sampler import RandomSampler
import cv2 as cv
import torch
import torchvision.transforms as T


DATASET_INFOS = {
    'mvtec': [
        ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile','toothbrush', 'transistor', 'wood', 'zipper'], 
        ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'], 
        ['carpet', 'grid', 'leather', 'tile', 'wood']
    ],  # all, obj, texture
    'mvtec_3d': [
        ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"], 
        ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"], 
        []
    ],
    'btad': [
        ["01", "02", "03"], 
        ["01", "03"], 
        ["02"]
    ]
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_image(path, resize = None):
    """
    Read an image from the specified path and optionally resize it.
    
    Parameters:
        path (str): The path to the image file.
        resize (tuple, optional): Tuple containing the desired width and height to resize the image to.
    
    Returns:
        numpy.ndarray: The image data as a NumPy array.
    """
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if resize:
        img = cv.resize(img, dsize=resize)
    return img

def read_mask(path, resize = None):
    """
    Read an mask from the specified path and optionally resize it.
    
    Parameters:
        path (str): The path to the mask file.
        resize (tuple, optional): Tuple containing the desired width and height to resize the image to.
    
    Returns:
        numpy.ndarray: The image data as a NumPy array.
    """
    mask = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if resize:
        mask = cv.resize(mask, dsize=resize, interpolation=cv.INTER_NEAREST)
    return mask

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def inverse_test_transform(image):
    denormalized = image * torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1) + \
                   torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    img = denormalized * 255.0
    img = img.to(torch.uint8)
    return img.cpu().numpy().transpose(1, 2, 0)

class InfiniteSampler(RandomSampler):
    def __iter__(self):
        while True: yield from super().__iter__()

from dataset.base import CPRDataset