import numpy as np
import torch
import math
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
            src_dim: int,
            trg_dim: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            hidden_dim: int,
            batch_first: bool,
            dim_feedforward: int = 2048,
            dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_dim,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first = batch_first)
        self.src_encoder = nn.Linear(src_dim, hidden_dim)
        self.trg_encoder = nn.Linear(trg_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        self.outs_decoder1 = nn.Linear(hidden_dim, dim_feedforward)
        self.outs_decoder2 = nn.Linear(dim_feedforward, trg_dim)


    def forward(self,
            src: Tensor,
            trg: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor):
        src_enc = self.positional_encoding(self.src_encoder(src))
        trg_enc = self.positional_encoding(self.trg_encoder(trg))
        outs = self.transformer(src_enc, trg_enc, src_mask, tgt_mask)
        outs = nn.functional.relu(self.outs_decoder1(outs))
        return self.outs_decoder2(outs)
