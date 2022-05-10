# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool, backbone_ours = None):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if backbone_ours is not None:
            self.body_ours = IntermediateLayerGetter(backbone_ours, return_layers=return_layers)
            self.num_channels = 2*num_channels
        else:
            self.body_ours = None
            self.num_channels = num_channels

    def forward(self, tensor_list):
        """supports both NestedTensor and torch.Tensor
        """

        if self.body_ours is not None:
            if isinstance(tensor_list, NestedTensor):
                xs = self.body(tensor_list.tensors)
                xs_ours = self.body_ours(tensor_list.tensors)
                out_ours: Dict[str, NestedTensor] = {}
                out: Dict[str, NestedTensor] = {}
                for ((name, x), (name_ours, x_ours))  in zip(xs.items(), xs_ours.items()):
                    m = tensor_list.mask
                    # print("MASK shape is: ", m.shape)
                    # print("X shape is: ", x.shape)
                    assert m is not None
                    mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                    # print("MASK shape is: ", mask.shape)
                    mask_ours = F.interpolate(m[None].float(), size=x_ours.shape[-2:]).to(torch.bool)[0]
                    out[name] = NestedTensor(torch.cat([x, x_ours], dim=1), mask)

            else:
                out = self.body(tensor_list)
                out_ours = self.body_ours(tensor_list)
                for key in out.keys():
                    feat_ours, mask_ours = out_ours[key].decompose()
                    feat, mask = out[key].decompose()
                    # print("MASK shape is: ", mask.shape)
                    # print("X shape is: ", feat.shape)
                    out[key] = NestedTensor(torch.cat([feat, feat_ours], dim=1), mask)

            # print()
            # print("expecting that out shape: ", out.shape, " and out_ours.shape: ", out_ours.shape, " are the same and should be concatenated alon the channels")
            # out = torch.cat([out, out_ours], dim=1)
            # print("concatted out is: ", out.shape)
        else:
            if isinstance(tensor_list, NestedTensor):
                xs = self.body(tensor_list.tensors)
                out: Dict[str, NestedTensor] = {}
                for name, x in xs.items():
                    m = tensor_list.mask
                    assert m is not None
                    mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                    out[name] = NestedTensor(x, mask)
            else:
                out = self.body(tensor_list)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 custom_model_path = "",
                 fused = False):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
        # load the SwAV pre-training model from the url instead of supervised pre-training model
        if fused:
            print("BE FUSION")
            print("swaV model")

            checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(state_dict, strict=False)

            backbone_ours = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
            print("LOADING OUR SAVED MODELLLLLL!!!!!!\n\n\n\n", custom_model_path)
            if custom_model_path == "":
                print("Who do you who do you think you are no custome model path but specifiying fuse?")
                raise Exception
            checkpoint = torch.load(custom_model_path,map_location="cpu")
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items() if "fc" not in k}
            # print(state_dict)
            backbone_ours.load_state_dict(state_dict, strict=False)
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

            super().__init__(backbone, train_backbone, num_channels, return_interm_layers, backbone_ours)
        else:
            if custom_model_path == '':
                print("Basic pretrained resnet")
                # checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
                # state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
                # backbone.load_state_dict(state_dict, strict=False)
                pass
            else:
                print("LOADING OUR SAVED MODELLLLLL!!!!!!\n\n\n\n", custom_model_path)
                checkpoint = torch.load(custom_model_path,map_location="cpu")
                state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items() if "fc" not in k}
                # print(state_dict)
                backbone.load_state_dict(state_dict, strict=False)
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
            super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        """supports both NestedTensor and torch.Tensor
        """
        if isinstance(tensor_list, NestedTensor):
            xs = self[0](tensor_list)
            out: List[NestedTensor] = []
            pos = []
            for name, x in xs.items():
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.tensors.dtype))
            return out, pos
        else:
            return list(self[0](tensor_list).values())



def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.custom_model_path, args.fused)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


#simple shoulder kaizen