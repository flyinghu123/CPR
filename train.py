from itertools import chain
from pprint import pformat
from tqdm import tqdm
import argparse
import os
import sys
import torch

from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F

from dataset import DATASET_INFOS, CPRDataset, InfiniteSampler
from models import create_model, MODEL_INFOS, CPR
from test import test
from utils import fix_seeds, save_dependencies_files


def get_args_parser():
    parser = argparse.ArgumentParser()
    # run
    parser.add_argument("--num-workers", type=int, default=8, help="num workers")
    parser.add_argument("-lp", "--log-path", type=str, default=None, help="log path")
    # data
    parser.add_argument("-dn", "--dataset-name", type=str, default="mvtec", choices=["mvtec", "mvtec_3d", "btad"], help="dataset name")
    parser.add_argument("-ss", "--scales", type=int, nargs="+", help="multiscale", default=[4, 8])
    parser.add_argument("-kn", "--k-nearest", type=int, default=10, help="k nearest")
    parser.add_argument("-na", "--n-anomaly", type=int, default=0, help="n test anomaly samples")
    parser.add_argument("-r", "--resize", type=int, default=320, help="image resize")
    parser.add_argument("-fd", "--foreground-dir", type=str, default=None, help="foreground dir")
    parser.add_argument("-rd", "--retrieval-dir", type=str, default='log/retrieval_mvtec_DenseNet_features.denseblock1_320', help="retrieval dir")
    parser.add_argument("-dd", "--data-dir", type=str, default='log/synthetic_mvtec_640_12000_True_jpg/', help="synthetic data dir")
    parser.add_argument("--sub-categories", type=str, nargs="+", default=None, help="sub categories", choices=list(chain(*[x[0] for x in list(DATASET_INFOS.values())])))
    # train
    parser.add_argument("-bs", "--batch-size", type=int, default=32)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=40000)
    # test
    parser.add_argument("-tps", "--test-per-steps", type=int, default=2000)
    parser.add_argument("-rs", "--region-sizes", type=int, nargs="+", help="local retrieval region size", default=[3, 1])
    parser.add_argument("--T", type=int, default=512)  # for image-level inference, DeSTSeg
    # model
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet', choices=list(MODEL_INFOS.keys()), help="pretrained model")
    return parser

class ContrastiveLoss(nn.Module):
    def __init__(self, p_alpha: float = 0.8, n_alpha: float = -0.2, exponent: int = 1) -> None:
        super().__init__()
        self.p_alpha = p_alpha
        self.n_alpha = n_alpha
        self.exponent = exponent

    def forward(self, query_features, ref_features, query_positive_points, ref_positive_points, positive_weight, query_negative_points, ref_negative_points, negative_weight):
        query_positive_descriptors = F.grid_sample(query_features, query_positive_points[:, :, None], align_corners=False)[:, :, :, 0]  # b x d x n
        ref_positive_descriptors = F.grid_sample(ref_features, ref_positive_points[:, :, None], align_corners=False)[:, :, :, 0]  # b x d x n
        pos_loss = torch.pow(torch.clamp(self.p_alpha - F.cosine_similarity(query_positive_descriptors, ref_positive_descriptors), min=0) * positive_weight, self.exponent).sum() / max(1, (positive_weight > 0).sum())
        query_negative_descriptors = F.grid_sample(query_features, query_negative_points[:, :, None], align_corners=False)[:, :, :, 0]  # b x d x n
        ref_negative_descriptors = F.grid_sample(ref_features, ref_negative_points[:, :, None], align_corners=False)[:, :, :, 0]  # b x d x n
        neg_loss = torch.pow(torch.clamp(F.cosine_similarity(query_negative_descriptors, ref_negative_descriptors) - self.n_alpha, min=0) * negative_weight, self.exponent).sum() / max(1, (negative_weight > 0).sum())
        return pos_loss, neg_loss
    
def train_one_step(model: CPR, batch, loss_fn):
    img, retrieval_img, positive_points, retrieval_positive_points, positive_weight, \
                    negative_points, retrieval_negative_points, negative_weight, \
                    anomaly_points, retrieval_anomaly_points, anomaly_weight = batch
    
    features_list, ori_features_list = model(torch.cat([img, retrieval_img]))
    features_list, retrieval_features_list = list(zip(*[torch.chunk(features, 2, 0) for features in features_list]))
    ori_features_list, retrieval_ori_features_list = list(zip(*[torch.chunk(features, 2, 0) for features in ori_features_list]))

    pos_loss = neg_loss = 0.
    # multiscale
    for features, retrieval_features, ori_features, retrieval_ori_features in zip(features_list, retrieval_features_list, ori_features_list, retrieval_ori_features_list):
        ori_negative_descriptors = F.grid_sample(ori_features, negative_points[:, :, None], align_corners=False)[:, :, :, 0]  # b x d x n
        retrieval_ori_negative_descriptors = F.grid_sample(retrieval_ori_features, retrieval_negative_points[:, :, None], align_corners=False)[:, :, :, 0]  # b x d x n
        _negative_weight = torch.square(ori_negative_descriptors - retrieval_ori_negative_descriptors).sum(1).detach() * (negative_weight > 0).float()
        _negative_weight = _negative_weight / (_negative_weight.sum(1, keepdim=True) + 1e-7) * negative_weight.sum(1, keepdim=True)
        # negative+anomaly
        _negative_points = torch.cat([negative_points, anomaly_points], 1)
        _retrieval_negative_points = torch.cat([retrieval_negative_points, retrieval_anomaly_points], 1)
        _negative_weight = torch.cat([_negative_weight, anomaly_weight], 1)
        _pos_loss, _neg_loss = loss_fn(features, retrieval_features, positive_points, retrieval_positive_points, positive_weight, _negative_points, _retrieval_negative_points, _negative_weight)
        pos_loss = pos_loss + _pos_loss
        neg_loss = neg_loss + _neg_loss
    return {
        'loss': pos_loss + neg_loss, 
        'pos_loss': pos_loss,
        'neg_loss': neg_loss
    }


def main(args):
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    logger.add(os.path.join(args.log_path, 'runtime.log'), mode='w')
    logger.info(f'run argv: {" ".join(sys.argv)}')
    logger.info('args: \n' + pformat(vars(args)))
    save_dependencies_files(os.path.join(args.log_path, 'src'))
    all_categories = DATASET_INFOS[args.dataset_name][0]
    sub_categories = all_categories if args.sub_categories is None else args.sub_categories
    assert all([sub_category in all_categories for sub_category in sub_categories]), f"{sub_categories} must all be in {all_categories}"
    model_info = MODEL_INFOS[args.pretrained_model]
    layers = [model_info['layers'][model_info['scales'].index(scale)] for scale in args.scales]
    for sub_category in sub_categories:
        logger_handler_id = logger.add(os.path.join(args.log_path, sub_category, 'runtime.log'), mode='w')
        seed_worker       = fix_seeds(66)
        model             = create_model(args.pretrained_model, layers).cuda().train()
        dataset           = CPRDataset(args.dataset_name, sub_category, args.resize, args.data_dir, args.scales, args.region_sizes, args.retrieval_dir, args.foreground_dir)
        writer            = SummaryWriter(os.path.join(args.log_path, sub_category), flush_secs=10)
        dataloader        = DataLoader(dataset, batch_size=args.batch_size, sampler=InfiniteSampler(dataset), num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)
        optimizer         = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
        loss_fn           = ContrastiveLoss(exponent=3).cuda()
        dataloader_iter   = iter(dataloader)
        for global_step in trange(1, args.steps+1, leave=False, desc=f'{sub_category} train'):
            batch  = [x.cuda() for x in next(dataloader_iter)]
            loss_d = train_one_step(model, batch, loss_fn)
            optimizer.zero_grad()
            loss_d['loss'].backward()
            optimizer.step()
            [writer.add_scalar(f"train/{k}", v.item(), global_step) for k, v in loss_d.items()]
            if global_step % args.test_per_steps == 0 or global_step == args.steps: 
                ret = test(model, dataset.train_fns, dataset.test_fns, dataset.retrieval_result, dataset.foreground_result, args.resize, args.region_sizes, dataset.root_dir, args.k_nearest, args.T)
                torch.save(model.state_dict(), os.path.join(args.log_path, sub_category, f'{global_step:05d}.pth'))
                logger.info(f'[{global_step}] {sub_category} test result: {" ".join([f"{k}: {v:.4f}" for k, v in ret.items()])}')
                [writer.add_scalar(f"test/{k}", v, global_step) for k, v in ret.items()]
                model.train()
        logger.remove(logger_handler_id)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = f'log/{args.dataset_name}_{args.pretrained_model}_[{",".join(map(str, args.scales))}]_[{",".join(map(str, args.region_sizes))}]_{args.k_nearest}_{args.n_anomaly}_{args.resize}_{args.foreground_dir is not None}'
    main(args)