from functools import partial
from typing import List, Any, Union, Dict, Optional

from timm.models import densenet
from torch.utils.hooks import RemovableHandle
from torchvision.models import efficientnet, resnet
from torchvision.models.inception import BasicConv2d
import torch
import torch.nn as nn


'''> https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/common.py#L277'''
class LastLayerToExtractReachedException(Exception):
    pass


class ForwardHook:
    def __init__(self, stop: bool = False):
        self._feature = None
        self.stop = stop
    
    def __call__(self, module, input, output):
        self._feature = output
        if self.stop: raise LastLayerToExtractReachedException()
    
    @property
    def feature(self):
        try: return self._feature
        finally: self._feature = None


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, layers: Union[List[str], Dict[str, str]]) -> None:
        super().__init__()
        self.shapes: Dict[str, torch.Size] = None
        self.backbone = backbone
        self.forward_hooks: List[ForwardHook] = []
        self.hook_handles: List[RemovableHandle] = []
        self.layers = isinstance(layers, dict) and layers or dict(zip(layers, layers))  # ordered dict
        for idx, layer in enumerate(layers):
            forward_hook = ForwardHook(idx == len(layers) - 1)  # last layer
            network_layer = backbone
            while "." in layer:
                extract_block, layer = layer.split(".", 1)
                network_layer = network_layer.__dict__["_modules"][extract_block]
            network_layer = network_layer.__dict__["_modules"][layer]
            self.hook_handles.append(network_layer.register_forward_hook(forward_hook))
            self.forward_hooks.append(forward_hook)
            
    def prune(self, input_size):
        example_inputs = [torch.randn(2, 3, input_size, input_size).to(next(self.backbone.parameters()).device)]
        forward_record = dict()
        def forward_hook(name, *args):
            forward_record[name] = 1
            while "." in name: 
                name = name.rsplit('.', 1)[0]
                forward_record[name] = 1
        hook_handles = [[m.register_forward_hook(partial(forward_hook, name)), forward_record.__setitem__(name, 0)][0] for name, m in self.backbone.named_modules()]
        with torch.no_grad(): self.shapes = {k: v.shape for k, v in self(*example_inputs).items()}
        [hook_handle.remove() for hook_handle in hook_handles]
        for name, v in forward_record.items():
            if v >= 1 or name == '': continue
            cur_module = self.backbone
            while "." in name:
                p_name, name = name.split('.', 1)
                if not hasattr(cur_module, p_name): break
                cur_module = getattr(cur_module, p_name)
            else: delattr(cur_module, name)
        return self
    
    def forward(self, *x):
        try: self.backbone(*x)
        finally: return dict(zip(self.layers.values(), map(lambda hook: hook.feature, self.forward_hooks)))


class BaseModel(nn.Module):
    def __init__(self, layers: List[str] = None, backbone_name: str = None, input_size: int = 320, frozen: bool = True):
        super().__init__()
        self.frozen = frozen
        self.layers = layers
        self.backbone_name = backbone_name
        # from torchvision.models.feature_extraction import create_feature_extractor
        self.feature_extractor = FeatureExtractor(self.load_backbone(), self.layers).prune(input_size)
        self.shapes = list(self.feature_extractor.shapes.values())
        if self.frozen:
            self.requires_grad_(False).eval()
            self.forward = torch.no_grad()(self.forward)

    def train(self, mode: bool = True):
        if not self.frozen: return super().train(mode)
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.eval()  # frozen batchnorm, dropout
        return self

    def forward(self, x) -> List[torch.Tensor]:
        return list(self.feature_extractor(x).values())
    
    def load_backbone(self):
        raise NotImplementedError("load backbone must be implemented")


class DenseNet(BaseModel):
    def __init__(self, layers: List[str] = ['features.denseblock1', 'features.denseblock2'], backbone_name: str = 'densenet201', **kwargs):
        super().__init__(layers, backbone_name, **kwargs)
    
    def load_backbone(self) -> densenet.DenseNet:
        return getattr(densenet, self.backbone_name)(pretrained=True)


class EfficientNet(BaseModel):
    def __init__(self, layers: List[str] = ['features.2', 'features.3'], backbone_name: str = 'efficientnet_b0', **kwargs):
        super().__init__(layers, backbone_name, **kwargs)
    
    def load_backbone(self) -> efficientnet.EfficientNet:
        return getattr(efficientnet, self.backbone_name)(pretrained=True)


class ResNet(BaseModel):
    def __init__(self, layers: List[str] = ['layer1', 'layer2'], backbone_name: str = 'resnet18', **kwargs):
        super().__init__(layers, backbone_name, **kwargs)
    
    def load_backbone(self) -> resnet.ResNet:
        return getattr(resnet, self.backbone_name)(pretrained=True)


class Inception(nn.Module):
    def __init__(self, in_channels: int = 192, out_channels: int = 256):
        super(Inception, self).__init__()
        assert out_channels % 4 == 0, 'out_channels must be divisible by 4'
        self.branch0 = BasicConv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, in_channels // 4, kernel_size=1, stride=1),
            BasicConv2d(in_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, out_channels // 4, kernel_size=1, stride=1),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)], 1)


class LocalRetrievalBranch(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels_list: List[int]) -> None:
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 192, kernel_size=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                Inception(192, 256),
                nn.Conv2d(256, out_channels, kernel_size=1)
            ) for in_channels, out_channels in zip(in_channels_list, out_channels_list)
        ])
    
    def forward(self, xs: List[torch.Tensor]):
        assert len(xs) == len(self.conv), f'input length must match conv num: {len(xs)} vs {len(self.conv)}'
        return [layer(x) for x, layer in zip(xs, self.conv)]
    
    
class CPR(nn.Module):
    def __init__(self, backbone: BaseModel, lrb: LocalRetrievalBranch) -> None:
        super().__init__()
        self.lrb = lrb
        self.backbone = backbone

    def forward(self, x):
        ori_features = self.backbone(x)
        return self.lrb(ori_features), ori_features


MODEL_INFOS = {
    'DenseNet': {'layers': ['features.denseblock1', 'features.denseblock2'], 'cls': DenseNet, 'scales': [4, 8]},
    'EfficientNet': {'layers': ['features.2', 'features.3'], 'cls': EfficientNet, 'scales': [4, 8]},
    'ResNet': {'layers': ['layer1', 'layer2'], 'cls': ResNet, 'scales': [4, 8]},
}


def create_model(model_name: str = 'DenseNet', layers: List[str] = ['features.denseblock1', 'features.denseblock2'], input_size: int = 320, output_dim: int = 384) -> CPR:
    backbone: BaseModel = MODEL_INFOS[model_name]['cls'](layers, input_size=input_size).eval()
    lrb = LocalRetrievalBranch([shape[1] for shape in backbone.feature_extractor.shapes.values()], [output_dim] * len(layers))
    return CPR(backbone, lrb)