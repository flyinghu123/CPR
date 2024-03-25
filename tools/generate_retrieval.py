from glob import glob
from itertools import chain
from pprint import pformat
import argparse
import json
import os
import sys
sys.path.append('./')

from loguru import logger
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch

from dataset import DATASET_INFOS, test_transform, read_image
from models import MODEL_INFOS, BaseModel
from models.grb import get_grb
from utils import COLORS, save_dependencies_files, fix_seeds


@torch.no_grad()
def gen_retrieval(save_path, dataset_name, model_name, layer, resize, knn, vis):
    device = torch.device('cuda')
    logger.info(f'gen_retrieval')
    logger.info(f'save to {save_path}')
    logger.info(f'params: {dataset_name} {model_name} {layer} {resize} {knn} {vis}')
    assert os.path.exists(os.path.join('./data', dataset_name)), f'{dataset_name} not exists'
    dataset_info = DATASET_INFOS[dataset_name]
    for sub_category in dataset_info[0]:  # all
        fix_seeds(66)
        model: BaseModel = MODEL_INFOS[model_name]['cls']([layer], input_size=resize).to(device)
        model.eval()
        root_dir = os.path.join('./data', dataset_name, sub_category)
        logger.info(f'generate {sub_category}')
        cur_target_save_path = os.path.join(save_path, sub_category)
        os.makedirs(cur_target_save_path, exist_ok=True)
        train_image = {}
        train_ks = []
        train_image_fns = sorted(glob(os.path.join(root_dir, 'train/*/*')))
        train_features = torch.zeros(len(train_image_fns), *model.shapes[0][1:], device=device)
        for i, fn in enumerate(tqdm(train_image_fns, desc='extract train features', leave=False)):
            assert os.path.exists(fn), f'{fn} not exists'
            k = os.path.relpath(fn, root_dir)
            train_ks.append(k)
            image = read_image(fn, (resize, resize))
            image_t = test_transform(image)
            feature = model(image_t[None].to(device))[0]
            train_features[i:i+1] = feature.detach()
            if vis:
                train_image[k] = image
        logger.info('predict retrieval')
        grb = get_grb(train_features)
        retrieval_result = {}
        train_code_img = {}
        for i, k in enumerate(tqdm(train_ks, desc='retreival train data', leave=False)):
            histogram, code = grb(train_features[i:i+1], return_code=True)
            retrieval_ks = [train_ks[retrieval_idx] for retrieval_idx in grb.retrieval(histogram)[1:]]  # exclude oneself
            retrieval_result[k] = retrieval_ks
            # save
            cur_save_dir = os.path.dirname(os.path.join(cur_target_save_path, k))
            os.makedirs(cur_save_dir, exist_ok=True)
            cur_image_name = os.path.basename(k).split('.', 1)[0]
            if vis:
                train_code_img[k] = (COLORS[cv.resize(code[0, 0].cpu().numpy(), (resize, resize), interpolation=cv.INTER_NEAREST)] * 255.).astype(np.uint8)
                image = np.concatenate([train_image[k]] + [train_image[rk] for rk in retrieval_ks[:knn]], 1)
                image = cv.resize(image, (resize*2, resize*2//knn))
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(cur_save_dir, f'r_{cur_image_name}.png'), image)
        for fn in tqdm(sorted(glob(os.path.join(root_dir, 'test/*/*'))), desc='retreival test data', leave=False):
            assert os.path.exists(fn), f'{fn} not exists'
            k = os.path.relpath(fn, root_dir)
            image = read_image(fn, (resize, resize))
            image_t = test_transform(image)
            feature = model(image_t[None].to(device))[0]
            histogram, code = grb(feature, return_code=True)
            retrieval_ks = [train_ks[retrieval_idx] for retrieval_idx in grb.retrieval(histogram)]
            retrieval_result[k] = retrieval_ks
            # save
            cur_save_dir = os.path.dirname(os.path.join(cur_target_save_path, k))
            os.makedirs(cur_save_dir, exist_ok=True)
            cur_image_name = os.path.basename(k).split('.', 1)[0]
            if vis:
                image = np.concatenate([image] + [train_image[rk] for rk in retrieval_ks[:knn]], 1)
                code_image = np.concatenate([(COLORS[cv.resize(code[0, 0].cpu().numpy(), (resize, resize), interpolation=cv.INTER_NEAREST)] * 255.).astype(np.uint8)] + [train_code_img[rk] for rk in retrieval_ks[:knn]], 1)
                image = np.concatenate([image, code_image], axis=0)
                image = cv.resize(image, (resize*2, resize*4//knn))
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(cur_save_dir, f'r_{cur_image_name}.png'), image)
        torch.save(grb, os.path.join(cur_target_save_path, f'grb.pth'))
        with open(os.path.join(cur_target_save_path, r'r_result.json'), 'w') as f:
            json.dump(retrieval_result, f, indent=4)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("-lp", "--log-path", type=str, default=None, help="log path")
    # data
    parser.add_argument("--dataset-name", type=str, default="mvtec", choices=["mvtec", "mvtec_3d", "btad"], help="dataset name")
    parser.add_argument("--resize", type=int, default=320, help="image resize")
    # vis
    parser.add_argument("--vis", action="store_true", help='save vis result')
    parser.add_argument("-k", "--k-nearest", type=int, default=10, help="k nearest")
    # model
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet', choices=list(MODEL_INFOS.keys()), help="pretrained model")
    parser.add_argument("--layer", type=str, default='features.denseblock1', choices=list(chain(*[v['layers']for k, v in MODEL_INFOS.items()])), help=f'feature layer, ' + ", ".join([f"{k}: {v['layers']}" for k, v in MODEL_INFOS.items()]))
    args = parser.parse_args()
    # check
    if args.layer not in MODEL_INFOS[args.pretrained_model]['layers']:
        parser.error(f'{args.layer} not in {MODEL_INFOS[args.pretrained_model]["layers"]}')
    if args.log_path is None:
        args.log_path = f'log/retrieval_{args.dataset_name}_{args.pretrained_model}_{args.layer}_{args.resize}'

    logger.add(os.path.join(args.log_path, 'runtime.log'))
    logger.info('args: \n' + pformat(vars(args)))
    assert torch.cuda.is_available(), f'cuda is not available'
    save_dependencies_files(os.path.join(args.log_path, 'src'))
    gen_retrieval(args.log_path, args.dataset_name, args.pretrained_model, args.layer, args.resize, args.k_nearest, args.vis)
    