# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # num_pos_feats is 128 from default user input
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # x is the image tensors
        x = tensor_list.tensors
        # mask is the image masks, at first, masks have zeros at the image part, and ones at the padded parts.
        mask = tensor_list.mask
        assert mask is not None
        # flip the mask values, ones at the image portions, and zeros at the padded portions.
        not_mask = ~mask
        # accumulate the image portion of mask values on y-axis, for each mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # accumulate the image portion of mask values on x-axis, for each mask
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # scale (min-max) values to between 0 to tau
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # 0 to tau because of sine and cosine later

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # outputs a range from 0 to 127
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # output a range from 1 to 10000^1

        pos_x = x_embed[:, :, :, None] / dim_t  # create the dim positional embedding (128) for each pixel in the mask
        pos_y = y_embed[:, :, :, None] / dim_t  # output shape is (b, h, w, 128) from (b, h, w)
        # create the interleaved positional encoding 
        # pos_x output shape is (b, h, w, 128)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # concat the pos_x and pos_y embedding, output shape is 
        # rearrange the dimensions to (batch_size, embedding_size, h, w)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    # args.hidden_dim is the Size of the embeddings (dimension of the transformer), default is 256
    # N_steps default is 256 // 2 = 128
    N_steps = args.hidden_dim // 2
    
    # args.position_embedding is the Type of positional embedding to use on top of the image features
    # default is sine
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
