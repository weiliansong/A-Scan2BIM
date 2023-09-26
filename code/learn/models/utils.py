import math
import torch


def pos_encode_1d(x, d_model=128):
    pe = torch.zeros(len(x), d_model, device=x.device)

    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    div_term = div_term.to(x.device)

    pos_w = x.clone().float().unsqueeze(1)

    pe[:, 0:d_model:2] = torch.sin(pos_w * div_term)
    pe[:, 1:d_model:2] = torch.cos(pos_w * div_term)

    return pe


def pos_encode_2d(x, y, d_model=128):
    assert len(x) == len(y)
    pe = torch.zeros(len(x), d_model, device=x.device)

    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    div_term = div_term.to(x.device)

    pos_w = x.clone().float().unsqueeze(1)
    pos_h = y.clone().float().unsqueeze(1)

    pe[:, 0:d_model:2] = torch.sin(pos_w * div_term)
    pe[:, 1:d_model:2] = torch.cos(pos_w * div_term)
    pe[:, d_model::2] = torch.sin(pos_h * div_term)
    pe[:, d_model + 1 :: 2] = torch.cos(pos_h * div_term)

    return pe


def get_geom_feats(coords):
    (bs, N, _) = coords.shape
    _coords = coords.reshape(bs * N, -1)
    _geom_enc_a = pos_encode_2d(x=_coords[:, 0], y=_coords[:, 1])
    _geom_enc_b = pos_encode_2d(x=_coords[:, 2], y=_coords[:, 3])
    _geom_feats = torch.cat([_geom_enc_a, _geom_enc_b], dim=-1)
    geom_feats = _geom_feats.reshape(bs, N, -1)

    return geom_feats
