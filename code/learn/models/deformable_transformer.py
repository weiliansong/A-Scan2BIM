import copy
import einops
import torch
import numpy as np
from torch import nn, Tensor
from models.ops.modules import MSDeformAttn
import torch.nn.functional as F
from models.utils import get_geom_feats


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


class DeformableAttnDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        deform_type="DETR",
        num_samples=None,
        pool_type=None,
    ):
        super().__init__()
        self.deform_type = deform_type
        self.n_levels = n_levels
        self.num_samples = num_samples
        self.pool_type = pool_type

        print("Attention layer deform type: %s" % deform_type)
        if deform_type == "DETR_dense":
            print("Attention layer num samples: %d" % num_samples)
            print("Attention layer pool_type: %s" % pool_type)

        # cross attention
        if deform_type in ["DETR", "DETR_dense"]:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        else:
            raise Exception("Unknown type of deformable attention!")

        # self.linear0 = nn.Linear(d_model * 3, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        edge_coords,
        image_size,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        key_padding_mask=None,
    ):
        # cross attention
        if self.deform_type == "DETR":
            ref_in = ref_out = None
            tgt2 = self.cross_attn(
                self.with_pos_embed(tgt, query_pos),
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
            )

        elif self.deform_type == "DETR_dense":
            assert len(edge_coords) == 1
            _edge_coords = edge_coords.cpu().numpy()
            x0 = _edge_coords[0, :, 0]
            y0 = _edge_coords[0, :, 1]
            x1 = _edge_coords[0, :, 2]
            y1 = _edge_coords[0, :, 3]

            # these are the provided reference points
            # we offset them inside the attention module
            k = self.num_samples
            ref_in = np.stack(
                [
                    np.linspace(x0, x1, k),
                    np.linspace(y0, y1, k),
                ],
            )
            ref_in = einops.rearrange(ref_in, "d k n -> n k d")

            # we have multiple levels
            ref_in = einops.repeat(ref_in, "n k d -> b (n k) l d", b=1, l=4)
            ref_in = torch.tensor(ref_in, device=tgt.device, dtype=tgt.dtype)

            # we also inject our sampling locations, so it can learn to vary
            _tgt = einops.repeat(tgt, "b n d -> b (n k) d", k=k)
            _query_pos = einops.repeat(query_pos, "b n d -> b (n k) d", k=k)

            ref_xy = ref_in[:, :, 0].clone()
            (h, w) = image_size
            ref_xy[..., 0] *= w
            ref_xy[..., 1] *= h
            ref_xyxy = torch.cat([ref_xy, ref_xy], dim=2)
            ref_pos = get_geom_feats(ref_xyxy)

            new_tgt = _tgt + _query_pos
            # new_tgt = torch.cat([_tgt, _query_pos, ref_pos], dim=-1)
            # new_tgt = self.linear0(new_tgt)

            tgt2, ref_out = self.cross_attn(
                new_tgt,
                ref_in,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                return_sample_locs=True,
            )
            tgt2 = einops.reduce(tgt2, "b (n k) d -> b n d", self.pool_type, k=k)

            ref_in = einops.rearrange(ref_in, "b (n k) l d -> b n k l d", k=k)
            ref_out = einops.rearrange(ref_out, "b (n k) h l p d -> b n k h l p d", k=k)

        elif self.deform_type == "DAT":
            raise Exception("Nope")
            assert (not src_padding_mask.any()) and (len(src_padding_mask) == 1)
            tgt2, pos, ref_out = self.cross_attn(
                edge_feats=tgt,
                edge_coords=edge_coords,
                edge_masks=key_padding_mask,
                lvls_feats=src,
                lvls_spatial_shapes=src_spatial_shapes,
                lvls_start_index=level_start_index,
            )

        else:
            raise Exception

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, ref_in, ref_out


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        deform_type=None,
        num_samples=None,
        pool_type=None,
    ):
        super().__init__()
        self.deform_type = deform_type
        self.num_samples = num_samples
        self.pool_type = pool_type

        print("Decoder layer deform type: %s" % deform_type)
        print("Decoder layer num samples: %d" % num_samples)
        print("Decoder layer pool_type: %s" % pool_type)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # self.linear0 = nn.Linear(d_model * 3, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        edge_coords,
        image_size,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        key_padding_mask=None,
        get_image_feat=True,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            key_padding_mask=key_padding_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        ref_in = ref_out = reference_points

        if get_image_feat:
            # cross attention
            if self.deform_type == "DETR":
                ref_in = ref_out = None
                tgt2 = self.cross_attn(
                    self.with_pos_embed(tgt, query_pos),
                    reference_points,
                    src,
                    src_spatial_shapes,
                    level_start_index,
                    src_padding_mask,
                )

            elif self.deform_type == "DETR_dense":
                assert len(edge_coords) == 1
                _edge_coords = edge_coords.cpu().numpy()
                x0 = _edge_coords[0, :, 0]
                y0 = _edge_coords[0, :, 1]
                x1 = _edge_coords[0, :, 2]
                y1 = _edge_coords[0, :, 3]

                k = self.num_samples
                ref_in = np.stack(
                    [
                        np.linspace(x0, x1, k),
                        np.linspace(y0, y1, k),
                    ],
                )
                ref_in = einops.rearrange(ref_in, "d k n -> n k d")

                # we have multiple levels
                ref_in = einops.repeat(ref_in, "n k d -> b (n k) l d", b=1, l=4)
                ref_in = torch.tensor(ref_in, device=tgt.device, dtype=tgt.dtype)

                # we also inject our sampling locations, so it can learn to vary
                _tgt = einops.repeat(tgt, "b n d -> b (n k) d", k=k)
                _query_pos = einops.repeat(query_pos, "b n d -> b (n k) d", k=k)

                ref_xy = ref_in[:, :, 0].clone()
                (h, w) = image_size
                ref_xy[..., 0] *= w
                ref_xy[..., 1] *= h
                ref_xyxy = torch.cat([ref_xy, ref_xy], dim=2)
                ref_pos = get_geom_feats(ref_xyxy)

                new_tgt = _tgt + _query_pos
                # new_tgt = torch.cat([_tgt, _query_pos, ref_pos], dim=-1)
                # new_tgt = self.linear0(new_tgt)

                tgt2, ref_out = self.cross_attn(
                    new_tgt,
                    ref_in,
                    src,
                    src_spatial_shapes,
                    level_start_index,
                    src_padding_mask,
                    return_sample_locs=True,
                )
                tgt2 = einops.reduce(tgt2, "b (n k) d -> b n d", self.pool_type, k=k)
                ref_in = einops.rearrange(ref_in, "b (n k) l d -> b n k l d", k=k)
                ref_out = einops.rearrange(
                    ref_out, "b (n k) h l p d -> b n k h l p d", k=k
                )

            else:
                raise Exception("Unknown deform type")

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, ref_in, ref_out


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, return_intermediate=False, with_sa=True
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.with_sa = with_sa

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        edge_coords=None,
        image_size=None,
        src_padding_mask=None,
        key_padding_mask=None,
        get_image_feat=True,
    ):
        output = tgt

        intermediate = []
        all_ref_in = []
        all_ref_out = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            if self.with_sa:
                output, ref_in, ref_out = layer(
                    output,
                    query_pos,
                    edge_coords,
                    image_size,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    src_padding_mask,
                    key_padding_mask,
                    get_image_feat,
                )
            else:
                output, ref_in, ref_out = layer(
                    output,
                    query_pos,
                    edge_coords,
                    image_size,
                    reference_points_input,
                    src,
                    src_spatial_shapes,
                    src_level_start_index,
                    src_padding_mask,
                    key_padding_mask,
                )

            # intermediate.append(output)
            all_ref_in.append(ref_in)
            all_ref_out.append(ref_out)

        # if self.return_intermediate:
        #     return torch.stack(intermediate), torch.stack(all_ref_ins)

        # return output, torch.stack(all_ref_in), torch.stack(all_ref_out)
        return output, all_ref_in, all_ref_out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
