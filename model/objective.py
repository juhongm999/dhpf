r"""Training objectives of DHPF"""
import math

import torch.nn.functional as F
import torch

from .base.geometry import Correlation
from .base.norm import Norm


class Objective:
    r"""Provides training objectives of DHPF"""
    @classmethod
    def initialize(cls, target_rate, alpha):
        cls.softmax = torch.nn.Softmax(dim=1)
        cls.target_rate = target_rate
        cls.alpha = alpha
        cls.eps = 1e-30

    @classmethod
    def weighted_cross_entropy(cls, correlation_matrix, easy_match, hard_match, batch):
        r"""Computes sum of weighted cross-entropy values between ground-truth and prediction"""
        loss_buf = correlation_matrix.new_zeros(correlation_matrix.size(0))
        correlation_matrix = Norm.unit_gaussian_normalize(correlation_matrix)

        for idx, (ct, thres, npt) in enumerate(zip(correlation_matrix, batch['pckthres'], batch['n_pts'])):

            # Hard (incorrect) match
            if len(hard_match['src'][idx]) > 0:
                cross_ent = cls.cross_entropy(ct, hard_match['src'][idx], hard_match['trg'][idx])
                loss_buf[idx] += cross_ent.sum()

            # Easy (correct) match
            if len(easy_match['src'][idx]) > 0:
                cross_ent = cls.cross_entropy(ct, easy_match['src'][idx], easy_match['trg'][idx])
                smooth_weight = (easy_match['dist'][idx] / (thres * cls.alpha)).pow(2)
                loss_buf[idx] += (smooth_weight * cross_ent).sum()

            loss_buf[idx] /= npt

        return torch.mean(loss_buf)

    @classmethod
    def cross_entropy(cls, correlation_matrix, src_match, trg_match):
        r"""Cross-entropy between predicted pdf and ground-truth pdf (one-hot vector)"""
        pdf = cls.softmax(correlation_matrix.index_select(0, src_match))
        prob = pdf[range(len(trg_match)), trg_match]
        cross_ent = -torch.log(prob + cls.eps)

        return cross_ent

    @classmethod
    def information_entropy(cls, correlation_matrix, rescale_factor=4):
        r"""Computes information entropy of all candidate matches"""
        bsz = correlation_matrix.size(0)

        correlation_matrix = Correlation.mutual_nn_filter(correlation_matrix)

        side = int(math.sqrt(correlation_matrix.size(1)))
        new_side = side // rescale_factor

        trg2src_dist = correlation_matrix.view(bsz, -1, side, side)
        src2trg_dist = correlation_matrix.view(bsz, side, side, -1).permute(0, 3, 1, 2)

        # Squeeze distributions for reliable entropy computation
        trg2src_dist = F.interpolate(trg2src_dist, [new_side, new_side], mode='bilinear', align_corners=True)
        src2trg_dist = F.interpolate(src2trg_dist, [new_side, new_side], mode='bilinear', align_corners=True)

        src_pdf = Norm.l1normalize(trg2src_dist.view(bsz, -1, (new_side * new_side)))
        trg_pdf = Norm.l1normalize(src2trg_dist.view(bsz, -1, (new_side * new_side)))

        src_pdf[src_pdf == 0.0] = cls.eps
        trg_pdf[trg_pdf == 0.0] = cls.eps

        src_ent = (-(src_pdf * torch.log2(src_pdf)).sum(dim=2)).view(bsz, -1)
        trg_ent = (-(trg_pdf * torch.log2(trg_pdf)).sum(dim=2)).view(bsz, -1)
        score_net = (src_ent + trg_ent).mean(dim=1) / 2

        return score_net.mean()

    @classmethod
    def layer_selection_loss(cls, layer_sel):
        r"""Encourages model to select each layer at a certain rate"""
        return (layer_sel.mean(dim=0) - cls.target_rate).pow(2).sum()
