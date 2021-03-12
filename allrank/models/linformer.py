#from allrank.models.transformer import *
import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from allrank.models.transformer import clones
from allrank.models.transformer import attention


class LinformerMultiHeadAttention(nn.Module):
    """
    Linformer Multi-headed attention block.
    """
    def __init__(self, h, d_model, seq_len,  proj_size, dropout=0.1):
        """
        :param h: number of attention heads
        :param d_model: input/output dimensionality
        :param proj_size: size to project the sequence length
        :param dropout: dropout probability
        """
        super(LinformerMultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        #Initialization is done later
        self.proj = nn.Parameter(torch.zeros(seq_len, proj_size))
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, distillation = False):
        """
        Forward pass through the multi-head attention block.
        :param query: query set of shape [batch_size, slate_size, self.d_model]
        :param key: key set of shape [batch_size, slate_size, self.d_model]
        :param value: value set of shape [batch_size, slate_size, self.d_model]
        :param mask: padding mask of shape [batch_size, slate_length] WRONG [batch_size, 1, slate_length]
        :return: output of shape [batch_size, slate_size, self.d_model]
        """

        if mask is not None:
            #same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 0) Linformer: we preemptively project the queries and the values

        proj_function = lambda args : torch.einsum("bsd,sp->bpd", *args)

        key, value = map(proj_function, zip((query, value), (self.proj, self.proj)))

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = \
            [linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn, scores = attention(query, key, value, mask=mask.permute(0, 1, 3, 2), dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if distillation:
            return self.linears[-1](x), scores
        return self.linears[-1](x)


