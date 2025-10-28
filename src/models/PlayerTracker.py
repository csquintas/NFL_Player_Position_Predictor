"""Transformer-based trajectory predictor for tracked NFL players."""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEncoder(nn.Module):
    """Additive sinusoidal PE (batch_first), parameter free encoding."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, d], make it so it saves to model state dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, d]
        return x + self.pe[:, :x.size(1), :]
    

class PlayerTrackerTransformer(nn.Module):
    """
    Encoder-decoder baseline:
      - Encoder: same as your history encoder (returns H=[B,F,d_model]).
      - Decoder: T learned query tokens, causal self-attn + cross-attn to H.
      - Head predicts deltas; add to last_xy.
    """
    def __init__(self, d_in, T_out, d_model=128, nhead=4, nlayers_enc=2, nlayers_dec=2, ff_proj_mult=4, dropout=0.1):
        """
        Args:
            d_in: # of features per frame
            T_out: # of frames to predict
        """
        super().__init__()
        self.T_out = T_out
        
        # Encoder
        self.in_proj = nn.Linear(d_in, d_model)
        self.pe_enc  = SinusoidalPositionEncoder(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_proj_mult*d_model, dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers_enc)

        # Decoder
        # Use the approach from DETR, that has learnable queries for each predicted frame
        # Their paper states that this is better for continuous outputs that are structure
        # Shouldn't act like a word seq2seq (test this)
        self.query_tokens = nn.Parameter(torch.randn(T_out, d_model))  # [T,d]
        self.pe_dec  = SinusoidalPositionEncoder(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ff_proj_mult*d_model, dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=nlayers_dec)

        self.output_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))

    def encode(self, hist, padded_frame_mask):
        """Encode historical player states into contextual embeddings.

        Args:
            hist: Tensor shaped `[B, F, d_in]` containing normalized input histories.
            padded_frame_mask: Boolean key padding mask of shape `[B, F]` where True marks padding.

        Returns:
            Tensor `[B, F, d_model]` of per-frame encoded representations.
        """
        x = self.in_proj(hist)                         # [B,F,d]
        x = self.pe_enc(x)                             # [B,F,d] 
        encoded_prethrow = self.encoder(x, src_key_padding_mask=padded_frame_mask)  # [B,F,d]
        return encoded_prethrow

    def forward(self, hist, padded_frame_mask, last_xy):
        """Autoregressively predict future XY coordinates.

        Args:
            hist: Batch of history tensors `[B, F, d_in]`(batches, frames, # of features).
            padded_frame_mask: Key padding mask `[B, F]` with True for padded positions.
            last_xy: Tensor `[B, 2]` containing last position before throwing.

        Returns:
            Tensor `[B, T_out, 2]` with absolute predicted XY coordinates.
        
        Notes:
            B = Batch Size
            F = # of frames in batch frame history
            T = # of frames in batch's target (to predict)
        """
        B = hist.size(0)
        encoded_prethrow = self.encode(hist, padded_frame_mask) # [B,F,d]

        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B,T,d]
        queries = self.pe_dec(queries)

        T = queries.size(1) 
        causal = torch.triu(torch.ones(T, T, device=queries.device, dtype=torch.bool), diagonal=1)

        # decode with cross-attention to encoded prethrow
        decoded_postsnap = self.decoder(tgt=queries, memory=encoded_prethrow, tgt_mask=causal)    # [B,T,d]
        deltas = self.output_head(decoded_postsnap)           # [B,T,2]
        return deltas + last_xy[:, None, :]                   # absolute XY
    


