# coding=utf-8
from json import encoder
from sys import flags
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.mlp import MLP

from models.deformable_transformer_full import (
    DeformableTransformerEncoderLayer,
    DeformableTransformerEncoder,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    DeformableAttnDecoderLayer,
)
from models.ops.modules import MSDeformAttn
from models.corner_models import PositionEmbeddingSine
from models.unet import ResNetBackbone
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import torch.nn.functional as F

# from utils.misc import NestedTensor
from models.utils import pos_encode_1d, pos_encode_2d

from shapely.geometry import Point, LineString, box, MultiLineString
from shapely import affinity


def unnormalize_edges(edges, normalize_param):
    lines = [((x0, y0), (x1, y1)) for (x0, y0, x1, y1) in edges]
    lines = MultiLineString(lines)

    # normalize so longest edge is 1000, and to not change aspect ratio
    (xfact, yfact) = normalize_param["scale"]
    xfact = 1 / xfact
    yfact = 1 / yfact
    lines = affinity.scale(lines, xfact=xfact, yfact=yfact, origin=(0, 0))

    # center edges around 0
    (xoff, yoff) = normalize_param["translation"]
    lines = affinity.translate(lines, xoff=-xoff, yoff=-yoff)

    # rotation
    if "rotation" in normalize_param.keys():
        angle = normalize_param["rotation"]
        lines = affinity.rotate(lines, -angle, origin=(0, 0))

    new_coords = [list(line.coords) for line in lines.geoms]
    new_coords = np.array(new_coords).reshape(-1, 4)

    return new_coords


def vis_data(data):
    import matplotlib.pyplot as plt

    bs = len(data["floor_name"])

    for b_i in range(bs):
        edge_coords = data["edge_coords"][b_i].cpu().numpy()
        edge_mask = data["edge_coords_mask"][b_i].cpu().numpy()
        edge_order = data["edge_order"][b_i].cpu().numpy()
        label = data["label"][b_i].cpu().numpy()

        for edge_i, (x0, y0, x1, y1) in enumerate(edge_coords):
            if not edge_mask[edge_i]:
                if label[edge_i] == 0:
                    assert edge_order[edge_i] == 0
                    plt.plot([x0, x1], [y0, y1], "-or")
                elif label[edge_i] == 3:
                    assert edge_order[edge_i] > 0
                    plt.plot([x0, x1], [y0, y1], "--oc")
                    plt.text(
                        (x0 + x1) / 2, (y0 + y1) / 2, str(edge_order[edge_i]), color="c"
                    )
                elif label[edge_i] == 1:
                    assert edge_order[edge_i] == 1
                    plt.plot([x0, x1], [y0, y1], "-oy")
                    plt.text(
                        (x0 + x1) / 2, (y0 + y1) / 2, str(edge_order[edge_i]), color="c"
                    )
                elif label[edge_i] == 2:
                    assert edge_order[edge_i] == 0
                    plt.plot([x0, x1], [y0, y1], "-og")
                    plt.text(
                        (x0 + x1) / 2, (y0 + y1) / 2, str(edge_order[edge_i]), color="c"
                    )
                else:
                    raise Exception

        # plt.title("%d" % label)
        plt.tight_layout()
        plt.show()
        plt.close()


class EdgeTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
    ):
        super(EdgeTransformer, self).__init__()
        self.d_model = d_model

        decoder_1_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
        )
        self.decoder_1 = DeformableTransformerDecoder(
            decoder_1_layer, num_decoder_layers, return_intermediate_dec, with_sa=True
        )

        self.type_embed = nn.Embedding(3, 128)
        self.input_head = nn.Linear(128 * 4, d_model)

        # self.final_head = MLP(
        #     input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2
        # )
        self.final_head = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_geom_feats(self, coords):
        (bs, N, _) = coords.shape
        _coords = coords.reshape(bs * N, -1)
        _geom_enc_a = pos_encode_2d(x=_coords[:, 0], y=_coords[:, 1])
        _geom_enc_b = pos_encode_2d(x=_coords[:, 2], y=_coords[:, 3])
        _geom_feats = torch.cat([_geom_enc_a, _geom_enc_b], dim=-1)
        geom_feats = _geom_feats.reshape(bs, N, -1)

        return geom_feats

    def forward(self, data):
        # vis_data(data)

        # obtain edge positional features
        edge_coords = data["edge_coords"]
        edge_mask = data["edge_coords_mask"]
        edge_order = data["edge_order"]

        dtype = edge_coords.dtype
        device = edge_coords.device

        # three types of nodes
        # 1. dummy node
        # 2. modelled node
        # 3. sequence node

        bs = len(edge_coords)

        # geom
        geom_feats = self.get_geom_feats(edge_coords)

        # node type (NEED TO BE BEFORE ORDER)
        max_order = 10
        node_type = torch.ones_like(edge_order)
        node_type[edge_order == 0] = 0
        node_type[edge_order == max_order] = 2
        type_feats = self.type_embed(node_type.long())

        # order
        order_feats = pos_encode_1d(edge_order.flatten())
        order_feats = order_feats.reshape(edge_order.shape[0], edge_order.shape[1], -1)

        # combine features and also add a dummy node
        edge_feats = torch.cat([geom_feats, type_feats, order_feats], dim=-1)
        edge_feats = self.input_head(edge_feats)

        # first do self-attention without any flags
        hs = self.decoder_1(
            edge_feats=edge_feats,
            geom_feats=geom_feats,
            image_feats=None,
            key_padding_mask=edge_mask,
            get_image_feat=False,
        )

        hs = self.final_head(hs)

        # return only the dummy node
        return hs


class EdgeTransformer2(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
    ):
        super(EdgeTransformer2, self).__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.type_embed = nn.Embedding(3, 128)
        self.input_head = nn.Linear(128 * 3 + 2, d_model)

        # self.final_head = MLP(
        #     input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2
        # )
        self.final_head = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MSDeformAttn):
        #         m._reset_parameters()

    def forward(self, data):
        # vis_data(data)

        # obtain edge positional features
        edge_coords = data["edge_coords"]
        edge_mask = data["edge_coords_mask"]
        modelled_coords = data["modelled_coords"]
        modelled_mask = data["modelled_coords_mask"]

        dtype = edge_coords.dtype
        device = edge_coords.device

        # first self-attention among the modelled edges
        (bs, N, _) = modelled_coords.shape
        _modelled_coords = modelled_coords.reshape(bs * N, -1)
        _geom_enc_a = pos_encode_2d(x=_modelled_coords[:, 0], y=_modelled_coords[:, 1])
        _geom_enc_b = pos_encode_2d(x=_modelled_coords[:, 2], y=_modelled_coords[:, 3])
        _geom_feats = torch.cat([_geom_enc_a, _geom_enc_b], dim=-1)
        modelled_geom = _geom_feats.reshape(bs, N, -1)
        modelled_geom = modelled_geom.permute([1, 0, 2])  # bs,S,E -> S,bs,E

        memory = self.encoder(modelled_geom, src_key_padding_mask=modelled_mask)

        # then SA and CA among the sequence edges
        (bs, N, _) = edge_coords.shape
        _edge_coords = edge_coords.reshape(bs * N, -1)
        _geom_enc_a = pos_encode_2d(x=_edge_coords[:, 0], y=_edge_coords[:, 1])
        _geom_enc_b = pos_encode_2d(x=_edge_coords[:, 2], y=_edge_coords[:, 3])
        _geom_feats = torch.cat([_geom_enc_a, _geom_enc_b], dim=-1)
        seq_geom = _geom_feats.reshape(bs, N, -1)
        seq_geom = seq_geom.permute([1, 0, 2])  # bs,S,E -> S,bs,E

        dummy_geom = torch.zeros(
            [1, bs, seq_geom.shape[-1]], dtype=dtype, device=device
        )
        seq_geom = torch.cat([dummy_geom, seq_geom], dim=0)

        # for edge positions, dummy node is 0, modelled edges are 0, sequence start at 1
        pos_inds = torch.arange(N + 1, dtype=dtype, device=device)
        pos_feats = pos_encode_1d(pos_inds)
        pos_feats = pos_feats.unsqueeze(1).repeat(1, bs, 1)

        # for edge type, dummy node is 0, modelled edges are 1, sequence start at 2
        flag_feats = torch.zeros([N + 1, bs], dtype=torch.int64, device=device)
        flag_feats[1:, :] = 1
        flag_feats = nn.functional.one_hot(flag_feats, num_classes=2)

        edge_feats = torch.cat([seq_geom, pos_feats, flag_feats], dim=-1)
        edge_feats = self.input_head(edge_feats)

        # need to pad the mask
        dummy_mask = torch.zeros([bs, 1], dtype=edge_mask.dtype, device=device)
        edge_mask = torch.cat([dummy_mask, edge_mask], dim=-1)

        # first do self-attention without any flags
        hs = self.decoder(
            tgt=edge_feats,
            memory=memory,
            tgt_key_padding_mask=edge_mask,
            memory_key_padding_mask=modelled_mask,
        )
        return self.final_head(hs[0])

        # hs = self.final_head(hs)

        # # return only the dummy node
        # return hs[:, 0]
