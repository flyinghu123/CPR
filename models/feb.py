from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import torch
import torch.nn as nn

class FMinMaxScaler:
    def __init__(self, ratio=0.01):
        self.ratio = ratio
        self.min = None
        self.max = None
        
    def fit(self, data):
        m0 = np.partition(data, int(data.shape[0] * self.ratio), axis=0)[int(data.shape[0] * self.ratio)-1]
        m1 = np.partition(data, -int(data.shape[0] * self.ratio), axis=0)[-int(data.shape[0] * self.ratio)]
        data = data[(data>=m0) & (data<=m1)]
        self.min = data.min(0).item()
        self.max = data.max(0).item()
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = np.clip(data, self.min, self.max)
        elif isinstance(data, torch.Tensor):
            data = torch.clamp(data, self.min, self.max)
        data = (data - self.min) / (self.max - self.min)
        return data


class ForegroundEstimateBranch(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1x1 = torch.nn.Conv2d(in_channels, 1, 1, 1).requires_grad_(False)
    
    def initialize_weights(self, lda: LinearDiscriminantAnalysis, normalizer: FMinMaxScaler):
        self.conv1x1.weight.data = torch.from_numpy(lda.coef_.T).float()[None, :, :, None] / torch.tensor(normalizer.max - normalizer.min).float()
        self.conv1x1.bias.data = (torch.from_numpy(lda.intercept_).float() - torch.tensor(normalizer.min).float()) / torch.tensor(normalizer.max - normalizer.min).float()
        return self
    
    @torch.no_grad()
    def forward(self, x):
        return torch.clamp(self.conv1x1(x), 0, 1)
    
    
def get_feb(train_features) -> ForegroundEstimateBranch:
    """
    Get the foreground estimate branch.

    Args:
        train_features (torch.Tensor): The train features of shape (B, C, H, W), 
            where B is the batch size, C is the number of channels, H is the height, and W is the width.
    Returns:
        ForegroundEstimateBranch
    """
    kmeans_f_num      = 50000
    lda_f_num         = 15000
    foreground_ratio  = 1/5
    background_ratio  = 3/80
    background_id_num = 1
    n_clusters        = 2
    random_state      = np.random.RandomState(66)
    lda               = LinearDiscriminantAnalysis()
    kmeans            = KMeans(n_clusters, n_init=10, random_state=random_state)
    normalizer        = FMinMaxScaler()
    # train_features: b x c x h x w
    B, C, H, W = train_features.shape
    image_features = train_features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512
    kmeans.fit(image_features.reshape(-1, C)[random_state.permutation(B*H*W)[:kmeans_f_num]])
    image_codes = kmeans.predict(image_features.reshape(-1, C)).reshape(B, H, W)
    # background
    background_mask = np.zeros((B, H, W), dtype=bool)
    background_mask[:, :int(background_ratio * H), :] = True
    background_mask[:, -int(background_ratio * H):, :] = True
    background_mask[:, int(background_ratio * H):-int(background_ratio * H), :int(background_ratio * W)] = True
    background_mask[:, int(background_ratio * H):-int(background_ratio * H), -int(background_ratio * W):] = True
    # foreground
    foreground_mask = np.zeros((B, H, W), dtype=bool)
    foreground_mask[:, int(image_codes.shape[1] / 2 - image_codes.shape[1] * foreground_ratio):int(image_codes.shape[1] / 2 + image_codes.shape[1] * foreground_ratio),
                    int(image_codes.shape[2] / 2 - image_codes.shape[2] * foreground_ratio):int(image_codes.shape[2] / 2 + image_codes.shape[2] * foreground_ratio)] = True
    # background id
    background_ids = np.eye(kmeans.n_clusters)[image_codes[background_mask]].sum(0).argsort()[kmeans.n_clusters-background_id_num:]
    
    # leave background id
    background_mask = background_mask & (np.stack([image_codes == background_id for background_id in background_ids]).sum(0) > 0)

    # remove background id
    foreground_mask = foreground_mask & (np.stack([image_codes != background_id for background_id in background_ids]).sum(0) >= len(background_ids))

    background_features = image_features[background_mask]
    foreground_features = image_features[foreground_mask]
    
    lda_f_num = min(lda_f_num, len(background_features), len(foreground_features))  # accelerate
    lda.fit(np.concatenate([
        background_features[random_state.permutation(len(background_features))[:lda_f_num]], 
        foreground_features[random_state.permutation(len(foreground_features))[:lda_f_num]]]), 
        np.concatenate([np.zeros((lda_f_num), dtype=int), np.ones((lda_f_num), dtype=int)]))
    normalizer.fit(lda.decision_function(image_features.reshape(-1, C)))
    return ForegroundEstimateBranch(C).initialize_weights(lda, normalizer)
