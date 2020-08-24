r"""Two different strategies of weak/strong supervisions"""
from abc import ABC, abstractmethod

import numpy as np
import torch

from model.objective import Objective


class SupervisionStrategy(ABC):
    r"""Different strategies for methods:"""
    @abstractmethod
    def get_image_pair(self, batch, *args):
        pass

    @abstractmethod
    def get_correlation(self, correlation_matrix):
        pass

    @abstractmethod
    def compute_loss(self, correlation_matrix, *args):
        pass


class StrongSupStrategy(SupervisionStrategy):
    def get_image_pair(self, batch, *args):
        r"""Returns (semantically related) pairs for strongly-supervised training"""
        return batch['src_img'], batch['trg_img']

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'ALL PAIRS' in a batch"""
        return correlation_matrix.clone().detach()

    def compute_loss(self, correlation_matrix, *args):
        r"""Strongly-supervised matching loss (L_{match})"""
        easy_match = args[0]['easy_match']
        hard_match = args[0]['hard_match']
        layer_sel = args[1]
        batch = args[2]

        loss_cre = Objective.weighted_cross_entropy(correlation_matrix, easy_match, hard_match, batch)
        loss_sel = Objective.layer_selection_loss(layer_sel)
        loss_net = loss_cre + loss_sel

        return loss_net


class WeakSupStrategy(SupervisionStrategy):
    def get_image_pair(self, batch, *args):
        r"""Forms positive/negative image paris for weakly-supervised training"""
        training = args[0]
        self.bsz = len(batch['src_img'])

        if training:
            shifted_idx = np.roll(np.arange(self.bsz), -1)
            trg_img_neg = batch['trg_img'][shifted_idx].clone()
            trg_cls_neg = batch['category_id'][shifted_idx].clone()
            neg_subidx = (batch['category_id'] - trg_cls_neg) != 0

            src_img = torch.cat([batch['src_img'], batch['src_img'][neg_subidx]], dim=0)
            trg_img = torch.cat([batch['trg_img'], trg_img_neg[neg_subidx]], dim=0)
            self.num_negatives = neg_subidx.sum()
        else:
            src_img, trg_img = batch['src_img'], batch['trg_img']
            self.num_negatives = 0

        return src_img, trg_img

    def get_correlation(self, correlation_matrix):
        r"""Returns correlation matrices of 'POSITIVE PAIRS' in a batch"""
        return correlation_matrix[:self.bsz].clone().detach()

    def compute_loss(self, correlation_matrix, *args):
        r"""Weakly-supervised matching loss (L_{match})"""
        layer_sel = args[1]
        loss_pos = Objective.information_entropy(correlation_matrix[:self.bsz])
        loss_neg = Objective.information_entropy(correlation_matrix[self.bsz:]) if self.num_negatives > 0 else 1.0
        loss_sel = Objective.layer_selection_loss(layer_sel)
        loss_net = (loss_pos / loss_neg) + loss_sel

        return loss_net
