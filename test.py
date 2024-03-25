from collections import defaultdict
from glob import glob
from itertools import chain
from tqdm import tqdm
import argparse
import json
import os
import torch

from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dataset import DATASET_INFOS, read_image, read_mask, test_transform
from metrics import compute_ap_torch, compute_pixel_auc_torch, compute_pro_torch, compute_image_auc_torch
from models import create_model, MODEL_INFOS, CPR
from utils import fix_seeds


def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("-dn", "--dataset-name", type=str, default="mvtec", choices=["mvtec", "mvtec_3d", "btad"], help="dataset name")
    parser.add_argument("-ss", "--scales", type=int, nargs="+", help="multiscale", default=[4, 8])
    parser.add_argument("-kn", "--k-nearest", type=int, default=10, help="k nearest")
    parser.add_argument("-r", "--resize", type=int, default=320, help="image resize")
    parser.add_argument("-fd", "--foreground-dir", type=str, default=None, help="foreground dir")
    parser.add_argument("-rd", "--retrieval-dir", type=str, default='log/retrieval_mvtec_DenseNet_features.denseblock1_320', help="retrieval dir")
    parser.add_argument("--sub-categories", type=str, nargs="+", default=None, help="sub categories", choices=list(chain(*[x[0] for x in list(DATASET_INFOS.values())])))
    parser.add_argument("--T", type=int, default=512)  # for image-level inference
    parser.add_argument("-rs", "--region-sizes", type=int, nargs="+", help="local retrieval region size", default=[3, 1])
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet', choices=list(MODEL_INFOS.keys()), help="pretrained model")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None, help="checkpoints")
    return parser

@torch.no_grad()
def test(model: CPR, train_fns, test_fns, retrieval_result, foreground_result, resize, region_sizes, root_dir, knn, T):
    model.eval()
    train_local_features = [torch.zeros((len(train_fns), out_channels, *shape[2:]), device='cuda') for shape, out_channels in zip(model.backbone.shapes, model.lrb.out_channels_list)]
    train_foreground_weights = []
    k2id = {}
    for idx, image_fn in enumerate(tqdm(train_fns, desc='extract train local features', leave=False)):
        k = os.path.relpath(image_fn, root_dir)
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        features_list, ori_features_list = model(image_t[None].cuda())
        for i, features in enumerate(features_list):
            train_local_features[i][idx:idx+1] = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
        if k in foreground_result:
            train_foreground_weights.append(torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda())
        k2id[k] = idx
    if train_foreground_weights:
        train_foreground_weights = torch.stack(train_foreground_weights)
    
    gts = []
    i_gts = []
    preds = defaultdict(list)
    for image_fn in tqdm(test_fns, desc='predict test data', leave=False):
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        k = os.path.relpath(image_fn, root_dir)
        image_name = os.path.basename(k)[:-4]
        anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
        mask_fn = os.path.join(root_dir, 'ground_truth', anomaly_name, image_name + '_mask.png')
        if os.path.exists(mask_fn):
            mask = read_mask(mask_fn, (resize, resize))
        else:
            mask = np.zeros((resize, resize))
        
        gts.append((mask > 127).astype(int))
        i_gts.append((mask > 127).sum() > 0 and 1 or 0)
        
        features_list, ori_features_list = model(image_t[None].cuda())
        features_list = [features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8) for features in features_list]
        retrieval_idxs = [k2id[retrieval_k] for retrieval_k in retrieval_result[k][:knn]]
        retrieval_features_list = [train_local_features[i][retrieval_idxs] for i in range(len(features_list))]
        
        scores = []
        assert len(features_list) == len(retrieval_features_list) == len(region_sizes)
        for features, retrieval_features, region_size in zip(features_list, retrieval_features_list, region_sizes):
            unfold = nn.Unfold(kernel_size=region_size, padding=region_size // 2)
            region_features = unfold(retrieval_features).reshape(retrieval_features.shape[0], retrieval_features.shape[1], -1, retrieval_features.shape[2], retrieval_features.shape[3])  # b x c x r^2 x h x w
            dist = (1 - (features[:, :, None] * region_features).sum(1))  # b x r^2 x h x w
            # fill position is set to a large value
            dist = dist / (unfold(torch.ones(1, 1, retrieval_features.shape[2], retrieval_features.shape[3], device=retrieval_features.device)).reshape(1, -1, retrieval_features.shape[2], retrieval_features.shape[3]) + 1e-8)
            score = dist.min(1)[0].min(0)[0]
            score = F.interpolate(
                score[None, None],
                size=(features_list[0].shape[2], features_list[0].shape[3]),
                mode="bilinear", align_corners=False
            )[0, 0]
            scores.append(score)
        score = torch.stack(scores).sum(0)
        score = F.interpolate(
            score[None, None],
            size=(mask.shape[0], mask.shape[1]),
            mode="bilinear", align_corners=False
        )[0, 0]
        if k in foreground_result:
            foreground_weight = torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
            foreground_weight = torch.cat([foreground_weight[None], train_foreground_weights[retrieval_idxs]]).max(0)[0]
            score = score * foreground_weight
        score_g = gaussian_blur(score[None], (33, 33), 4)[0]  # PatchCore
        det_score = torch.topk(score_g.flatten(), k=T)[0].sum()  # DeSTSeg
        preds['i'].append(det_score)
        preds['p'].append(score_g)
    gts = torch.from_numpy(np.stack(gts)).cuda()
    return {
        'pro': compute_pro_torch(gts, torch.stack(preds['p'])),
        'ap': compute_ap_torch(gts, torch.stack(preds['p'])),
        'pixel-auc': compute_pixel_auc_torch(gts, torch.stack(preds['p'])),
        'image-auc': compute_image_auc_torch(torch.tensor(i_gts).long().cuda(), torch.stack(preds['i'])),
    }

def main(args):
    all_categories, object_categories, texture_categories = DATASET_INFOS[args.dataset_name]
    sub_categories = DATASET_INFOS[args.dataset_name][0] if args.sub_categories is None else args.sub_categories
    assert all([sub_category in all_categories for sub_category in sub_categories]), f"{sub_categories} must all be in {all_categories}"
    model_info = MODEL_INFOS[args.pretrained_model]
    layers = [model_info['layers'][model_info['scales'].index(scale)] for scale in args.scales]
    for sub_category_idx, sub_category in enumerate(sub_categories):
        fix_seeds(66)
        model             = create_model(args.pretrained_model, layers).cuda()
        if args.checkpoints is not None:
            checkpoint_fn = args.checkpoints[0] if len(args.checkpoints) == 1 else args.checkpoints[sub_category_idx]
            if '{category}' in checkpoint_fn: checkpoint_fn = checkpoint_fn.format(category=sub_category)
            model.load_state_dict(torch.load(checkpoint_fn), strict=False)
        root_dir = os.path.join('./data', args.dataset_name, sub_category)
        train_fns = sorted(glob(os.path.join(root_dir, 'train/*/*')))
        test_fns = sorted(glob(os.path.join(root_dir, 'test/*/*')))
        with open(os.path.join(args.retrieval_dir, sub_category, 'r_result.json'), 'r') as f:
            retrieval_result = json.load(f)
        forground_result = {}
        if args.foreground_dir is not None and sub_category in object_categories:
            for fn in train_fns + test_fns:
                k = os.path.relpath(fn, root_dir)
                forground_result[k] = os.path.join(args.foreground_dir, sub_category, os.path.dirname(k), 'f_' + os.path.splitext(os.path.basename(k))[0] + '.npy')
        ret = test(model, train_fns, test_fns, retrieval_result, forground_result, args.resize, args.region_sizes, root_dir, args.k_nearest, args.T)
        print(f'================={sub_category}=================')
        print(ret)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)