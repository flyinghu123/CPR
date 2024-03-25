from typing import Tuple

from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn


def entropy_pytorch(pk, qk, dim=0):
    return (pk * torch.log(pk / qk)).sum(dim)


class Codebook(nn.Module):
    def __init__(self, n_clusters, in_channels) -> None:
        super().__init__()
        self.cluster_centers = nn.Parameter(torch.zeros(1, n_clusters, in_channels, 1, 1), requires_grad=False)
    
    def initialize_weights(self, kmeans: KMeans):
        self.cluster_centers.data = torch.from_numpy(kmeans.cluster_centers_).float()[None, :, :, None, None]
        
    def forward(self, x):
        return torch.square((x[:, None] - self.cluster_centers)).sum(2).argmin(1, keepdim=True)  # b x 1 x h x w


class BlockWiseHistogramEncoder(nn.Module):
    def __init__(self, S: int, input_size: int, n_clusters: int) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        block_size = (input_size + S - 1) // S
        padding_size = block_size * S - input_size
        self.unfold = nn.Unfold(kernel_size=(block_size, block_size), stride=block_size, padding=padding_size)
    
    def forward(self, x):
        x = self.unfold(x.float() + 1).long()  # b x S^2 x n, padding的id为0
        x = nn.functional.one_hot(x, self.n_clusters + 1).float().mean(1)
        return x
    

class GlobalRetrievalBranch(nn.Module):
    def __init__(self, bank_size, input_size, in_channels, n_clusters, S, d_method='kl', l_ratio=4/5) -> None:
        super().__init__()
        self.l_ratio = l_ratio
        self.d_method = d_method
        self.codebook = Codebook(n_clusters, in_channels)
        self.block_wise_histogram_encoder = BlockWiseHistogramEncoder(S, input_size, n_clusters)
        self.refs = nn.Parameter(torch.zeros(bank_size, S**2, n_clusters), requires_grad=False)
        self.register_buffer('_bank', torch.tensor(False))
    
    def initialize_weights(self, kmeans: KMeans):
        self.codebook.initialize_weights(kmeans)
        return self
    
    @torch.no_grad()
    def forward(self, x, return_code = False):
        code = self.codebook(x)
        x = self.block_wise_histogram_encoder(code)
        if return_code:
            return x, code
        return x
    
    def set_bank(self, refs):
        self.refs.data = refs
        self._bank = torch.tensor(True, device=refs.device)
    
    def retrieval(self, query):
        assert self._bank, f'GlobalRetrievalBranch must set bank before retrieval.'
        if self.d_method == 'kl':
            idx = torch.argsort(torch.sort(entropy_pytorch(
                        query + 1e-8, self.refs + 1e-8, -1
                    ), -1)[0][:, :int(query.shape[1] * self.l_ratio)].sum(-1))
        else:
            idx = torch.argsort(torch.sort(torch.norm(
                        query - self.refs, dim=-1
                    ), -1)[0][:, :int(query.shape[1] * self.l_ratio)].sum(-1))
        return idx
    
    
def get_grb(train_features) -> GlobalRetrievalBranch:
    """
    Get the global retrieval branch.

    Args:
        train_features (torch.Tensor): The train features of shape (B, C, H, W), 
            where B is the batch size, C is the number of channels, H is the height, and W is the width.

    Returns:
        GlobalRetrievalBranch
    """
    kmeans_f_num = 50000
    S            = 5
    n_clusters   = 12
    random_state = np.random.RandomState(66)
    kmeans       = KMeans(n_clusters, n_init=10, random_state=random_state)
    B, C, H, W = train_features.shape
    kmeans.fit(train_features.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, C)[random_state.permutation(B*H*W)[:kmeans_f_num]])
    grb = GlobalRetrievalBranch(B, H, C, n_clusters, S).initialize_weights(kmeans).to('cuda')
    grb.set_bank(torch.cat([grb(train_features[i:i+1]) for i in range(B)]))
    return grb
    