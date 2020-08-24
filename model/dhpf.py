"""Implementation of Dynamic Hyperpixel Flow"""
from functools import reduce
from operator import add

import torch.nn as nn
import torch

from .base.correlation import Correlation
from .base.geometry import Geometry
from .base.norm import Norm
from .base import resnet
from . import gating
from . import rhm


class DynamicHPF:
    r"""Dynamic Hyperpixel Flow (DHPF)"""
    def __init__(self, backbone, device, img_side=240):
        r"""Constructor for DHPF"""
        super(DynamicHPF, self).__init__()

        # 1. Backbone network initialization
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512, 1024,
                                1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            self.in_channels = [64, 256, 256, 256, 512, 512, 512, 512,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 1024, 1024, 1024, 1024, 1024,
                                1024, 1024, 2048, 2048, 2048]
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.backbone.eval()

        # 2. Dynamic layer gatings initialization
        self.learner = gating.GumbelFeatureSelection(self.in_channels).to(device)

        # 3. Miscellaneous
        self.relu = nn.ReLU()
        self.upsample_size = [int(img_side / 4)] * 2
        Geometry.initialize(self.upsample_size, device)
        self.rhm = rhm.HoughMatching(Geometry.rfs, torch.tensor([img_side, img_side]).to(device))

    # Forward pass
    def __call__(self, *args, **kwargs):
        # 1. Compute correlations between hyperimages
        src_img = args[0]
        trg_img = args[1]
        correlation_matrix, layer_sel = self.hyperimage_correlation(src_img, trg_img)

        # 2. Compute geometric matching scores to re-weight appearance matching scores (RHM)
        with torch.no_grad():  # no back-prop thru rhm due to memory issue
            geometric_scores = torch.stack([self.rhm.run(c.clone().detach()) for c in correlation_matrix], dim=0)
        correlation_matrix *= geometric_scores

        return correlation_matrix, layer_sel

    def hyperimage_correlation(self, src_img, trg_img):
        r"""Dynamically construct hyperimages and compute their correlations"""
        layer_sel = []
        correlation, src_norm, trg_norm = 0, 0, 0

        # Concatenate source & target images (B,6,H,W)
        # Perform group convolution (group=2) for faster inference time
        pair_img = torch.cat([src_img, trg_img], dim=1)

        # Layer 0
        with torch.no_grad():
            feat = self.backbone.conv1.forward(pair_img)
            feat = self.backbone.bn1.forward(feat)
            feat = self.backbone.relu.forward(feat)
            feat = self.backbone.maxpool.forward(feat)

            src_feat = feat.narrow(1, 0, feat.size(1) // 2).clone()
            trg_feat = feat.narrow(1, feat.size(1) // 2, feat.size(1) // 2).clone()

        # Save base maps
        base_src_feat = self.learner.reduction_ffns[0](src_feat)
        base_trg_feat = self.learner.reduction_ffns[0](trg_feat)
        base_correlation = Correlation.bmm_interp(base_src_feat, base_trg_feat, self.upsample_size)
        base_src_norm = Norm.feat_normalize(base_src_feat, self.upsample_size)
        base_trg_norm = Norm.feat_normalize(base_trg_feat, self.upsample_size)

        src_feat, trg_feat, lsel = self.learner(0, src_feat, trg_feat)
        if src_feat is not None and trg_feat is not None:
            correlation += Correlation.bmm_interp(src_feat, trg_feat, self.upsample_size)
            src_norm += Norm.feat_normalize(src_feat, self.upsample_size)
            trg_norm += Norm.feat_normalize(trg_feat, self.upsample_size)
        layer_sel.append(lsel)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            with torch.no_grad():
                res = feat
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)
                if bid == 0:
                    res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)
                feat += res

                src_feat = feat.narrow(1, 0, feat.size(1) // 2).clone()
                trg_feat = feat.narrow(1, feat.size(1) // 2, feat.size(1) // 2).clone()

            src_feat, trg_feat, lsel = self.learner(hid + 1, src_feat, trg_feat)
            if src_feat is not None and trg_feat is not None:
                correlation += Correlation.bmm_interp(src_feat, trg_feat, self.upsample_size)
                src_norm += Norm.feat_normalize(src_feat, self.upsample_size)
                trg_norm += Norm.feat_normalize(trg_feat, self.upsample_size)
            layer_sel.append(lsel)

            with torch.no_grad():
                feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        layer_sel = torch.stack(layer_sel).t()

        # If no layers are selected, select the base map
        if (layer_sel.sum(dim=1) == 0).sum() > 0:
            empty_sel = (layer_sel.sum(dim=1) == 0).nonzero().view(-1).long()
            if src_img.size(0) == 1:
                correlation = base_correlation
                src_norm = base_src_norm
                trg_norm = base_trg_norm
            else:
                correlation[empty_sel] += base_correlation[empty_sel]
                src_norm[empty_sel] += base_src_norm[empty_sel]
                trg_norm[empty_sel] += base_trg_norm[empty_sel]

        if self.learner.training:
            src_norm[src_norm == 0.0] += 0.0001
            trg_norm[trg_norm == 0.0] += 0.0001
        src_norm = src_norm.pow(0.5).unsqueeze(2)
        trg_norm = trg_norm.pow(0.5).unsqueeze(1)

        # Appearance matching confidence (p(m_a)): cosine similarity between hyperpimages
        correlation_ts = self.relu(correlation / (torch.bmm(src_norm, trg_norm) + 0.001)).pow(2)

        return correlation_ts, layer_sel

    def parameters(self):
        return self.learner.parameters()

    def state_dict(self):
        return self.learner.state_dict()

    def load_state_dict(self, state_dict):
        self.learner.load_state_dict(state_dict)

    def eval(self):
        self.learner.eval()

    def train(self):
        self.learner.train()
