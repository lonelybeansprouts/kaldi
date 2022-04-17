#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Label smoothing module."""

import torch
from torch import nn
from bfctc_pytorch import BfCtcLoss
from pynet.utils.common import IGNORE_ID


class LabelBFCTCLoss(nn.Module):
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
                 normalize_length: bool = False,
                 input_size: int = -1,
                 ctc_weight: float=1.0):
        """Construct an LabelSmoothingLoss object."""
        super(LabelBFCTCLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length
        self.bfctc = BfCtcLoss(size_average=True)
        self.ctc_weight = ctc_weight
        if input_size>0:
            self.output_layer = torch.nn.Linear(input_size, size)
        else:
            self.output_layer = None


    def forward(self, x: torch.Tensor, ce_target: torch.Tensor, ctc_target: torch.Tensor, ctc_acts_lens: torch.Tensor, ctc_target_lens: torch.Tensor) -> torch.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (torch.Tensor): prediction (batch, seqlen, class)
            ce_target (torch.Tensor):
                ce target signal masked with self.padding_id (batch, seqlen)
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

        ce_loss=torch.tensor([0.0], dtype=torch.float32).to(x.device)
        if (ce_target != None and self.ctc_weight < 1.0):
            assert x.size(1) == ce_target.size(1) #check seqlen
            batch_size = x.size(0)
            x = x.view(-1, self.size)
            ce_target = ce_target.view(-1)
            # use zeros_like instead of torch.no_grad() for true_dist,
            # since no_grad() can not be exported by JIT
            true_dist = torch.zeros_like(x)
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = ce_target == self.padding_idx  # (B,)
            total = len(ce_target) - ignore.sum().item()
            ce_target = ce_target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, ce_target.unsqueeze(1).long(), self.confidence)
            kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
            denom = total if self.normalize_length else batch_size
            ce_loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

        ctc_loss=torch.tensor([0.0], dtype=torch.float32)
        if (self.ctc_weight > 0.0):#run on cpu
            ctc_acts = ctc_acts.transpose(0, 1) #transpose to (T, batch, feat_dim) and must be before softmax
            ctc_acts = ctc_acts.cpu().float()
            ctc_target = torch.masked_select(ctc_target, ctc_target.ne(IGNORE_ID)).cpu().long()
            ctc_acts_lens = ctc_acts_lens.cpu().long()
            ctc_target_lens = ctc_target_lens.cpu().long()
            ctc_loss = self.bfctc(ctc_acts, ctc_target, ctc_acts_lens, ctc_target_lens)
        
        return (1-self.ctc_weight)*ce_loss, self.ctc_weight*ctc_loss.to(x.device)


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
