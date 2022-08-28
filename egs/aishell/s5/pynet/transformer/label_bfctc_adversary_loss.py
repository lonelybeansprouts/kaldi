#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Label smoothing module."""

import torch
from torch import nn
from bfctc_pytorch import BfCtcLoss
from pynet.utils.common import IGNORE_ID


class LabelBfCtcAdversaryLoss(nn.Module):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """
    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 adversary_weight: float,
                 normalize_length: bool = False,
                 input_size: int = -1):
        """Construct an LabelSmoothingLoss object."""
        super(LabelBfCtcAdversaryLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.adversary_weight = adversary_weight
        self.normalize_length = normalize_length
        self.bfctc = BfCtcLoss(size_average=True)
        if input_size>0:
            self.output_layer = torch.nn.Linear(input_size, size)
        else:
            self.output_layer = None


    def forward(self, x: torch.Tensor, 
                ctc_target_p1: torch.Tensor, 
                ctc_target_p2: torch.Tensor,
                ctc_acts_lens: torch.Tensor, 
                ctc_target_lens_p1: torch.Tensor,
                ctc_target_lens_p2: torch.Tensor):
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            ctc_target (torch.Tensor):
                ctc target signal masked with self.padding_id (batch, seqlen)
            ctc_acts_lens (torch.Tensor):
                (batch)
            ctc_target_lens (torch.Tensor):
                (batch)
        Returns:
            loss (torch.Tensor) : The KL loss, scalar float value
        """
        if self.output_layer != None:
            x = self.output_layer(x)
        ctc_acts = x
        ctc_acts = ctc_acts.transpose(0, 1) #transpose to (T, batch, feat_dim) and must be before softmax
        ctc_acts = ctc_acts.cpu().float()
        ctc_acts_lens = ctc_acts_lens.cpu().long()


        ctc_target_p1 = torch.masked_select(ctc_target_p1, ctc_target_p1.ne(IGNORE_ID)).cpu().long()
        ctc_target_lens_p1 = ctc_target_lens_p1.cpu().long()

        ctc_loss_p1 = self.bfctc(ctc_acts, ctc_target_p1, ctc_acts_lens, ctc_target_lens_p1)

        ctc_target_p2 = torch.masked_select(ctc_target_p2, ctc_target_p2.ne(IGNORE_ID)).cpu().long()
        ctc_target_lens_p2 = ctc_target_lens_p2.cpu().long()
        ctc_loss_p2 = self.bfctc(ctc_acts, ctc_target_p2, ctc_acts_lens, ctc_target_lens_p2)



        return ctc_loss_p1-self.adversary_weight*ctc_loss_p2, ctc_loss_p1, ctc_loss_p2


    def get_logits(self, x: torch.Tensor):
        if self.output_layer != None:
            x = self.output_layer(x)
        return x

    def get_probs(self, x: torch.Tensor):
        if self.output_layer != None:
            x = self.output_layer(x)
        return torch.nn.functional.softmax(x, dim=-1)

    def get_logprobs(self, x: torch.Tensor):
        if self.output_layer != None:
            x = self.output_layer(x)
        return torch.nn.functional.log_softmax(x, dim=-1)
