import time
import copy
from gc import collect
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, Tensor
from torch.autograd import Variable
from models.ops.modules import MSDeformAttn
from shapely.geometry import LineString, box
import torch.nn.functional as F
from utils.nn_utils import pos_encode_2d


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
    ):
        super().__init__()
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
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
        global_coords,
        image_data,
        image_size,
        edge_input_fc,
        gpu_id,
        img_side_len=256,
        interp_dist=128,
    ):
        crop_infos = []
        crop_coords = []
        reference_points = []
        all_src = []
        all_src_spatial_shapes = []
        all_level_start_index = []
        all_src_padding_mask = []

        (ax, ay, bx, by) = global_coords
        edge_shp = LineString([[ax, ay], [bx, by]])
        edge_len = edge_shp.length
        num_interp = np.ceil(edge_len / interp_dist).astype(int)

        for interp_i in range(num_interp + 1):  # +1 to include the end point
            # determine center of crop
            edge_center = edge_shp.interpolate(interp_i / num_interp, normalized=True)
            center_x, center_y = int(edge_center.x), int(edge_center.y)

            # determine crop boundary
            (img_h, img_w) = image_size
            crop_minx = max(center_x - img_side_len // 2, 0)
            crop_maxx = crop_minx + img_side_len
            crop_miny = max(center_y - img_side_len // 2, 0)
            crop_maxy = crop_miny + img_side_len

            if crop_maxx > img_w:
                crop_maxx = img_w
                crop_minx = crop_maxx - img_side_len

            if crop_maxy > img_h:
                crop_maxy = img_h
                crop_miny = crop_maxy - img_side_len

            assert ((crop_maxx - crop_minx) == img_side_len) and (
                (crop_maxy - crop_miny) == img_side_len
            )

            # determine the cropped line's coordinates
            crop_bbox = box(crop_minx, crop_miny, crop_maxx, crop_maxy)
            crop_line = crop_bbox.intersection(edge_shp)

            ((crop_ax, crop_ay), (crop_bx, crop_by)) = crop_line.coords
            crop_ax -= crop_minx
            crop_bx -= crop_minx
            crop_ay -= crop_miny
            crop_by -= crop_miny
            crop_coords.append([crop_ax, crop_ay, crop_bx, crop_by])

            # but the reference point is in the local coordinate space
            ref_x = (center_x - crop_minx) / img_side_len
            ref_y = (center_y - crop_miny) / img_side_len
            reference_points.append(torch.FloatTensor([ref_x, ref_y]).cuda())

            # save how we generate this crop
            crop_info = {
                "bbox": [crop_minx, crop_miny, crop_maxx, crop_maxy],
                "coords": [crop_ax, crop_ay, crop_bx, crop_by],
                "ref": [ref_x, ref_y],
            }
            crop_infos.append(crop_info)

            continue

            # visualize the coordinates
            if False:
                plt.imshow(image_data["image"] / 3)

                # plot the edge in global
                plt.plot([ax, bx], [ay, by], "-^y")

                # plot this crop's bounding box
                xx = [crop_minx, crop_maxx, crop_maxx, crop_minx, crop_minx]
                yy = [crop_miny, crop_miny, crop_maxy, crop_maxy, crop_miny]
                plt.plot(xx, yy, "-c")

                # plot the edge in local
                plt.plot(
                    [crop_ax + crop_minx, crop_bx + crop_minx],
                    [crop_ay + crop_miny, crop_by + crop_miny],
                    "-vg",
                )

                # plot the reference in local
                plt.plot(
                    ref_x * img_side_len + crop_minx,
                    ref_y * img_side_len + crop_miny,
                    "or",
                )

                plt.axis("off")
                plt.show()

            # figure out where the new crop would be in each feature level
            src_flatten = []
            mask_flatten = []
            spatial_shapes = []
            side_lens = [32, 16, 8, 4]

            for feat_l, (feat_h, feat_w) in enumerate(image_data["spatial_shapes"]):
                ratio_h = img_h / feat_h
                ratio_w = img_w / feat_w

                l_minx = (crop_minx / ratio_w).round().long()
                l_miny = (crop_miny / ratio_h).round().long()
                l_maxx = l_minx + side_lens[feat_l]
                l_maxy = l_miny + side_lens[feat_l]
                # l_maxx = (crop_maxx / ratio_w).round().long()
                # l_maxy = (crop_maxy / ratio_h).round().long()

                l_h = l_maxy - l_miny
                l_w = l_maxx - l_minx
                assert (l_h == l_w) and (l_h in side_lens)
                spatial_shapes.append(torch.tensor([l_h, l_w]).cuda())

                start = image_data["level_start_index"][feat_l]
                end = start + feat_h * feat_w
                assert end <= image_data["src_flatten"].shape[1]

                src = image_data["src_flatten"][:, start:end]
                src = src.reshape([1, feat_h, feat_w, -1])
                src_crop = src[:, l_miny:l_maxy, l_minx:l_maxx]
                src_flatten.append(src_crop.reshape(1, l_h * l_w, -1))

                mask = image_data["mask_flatten"][:, start:end]
                mask = mask.reshape([1, feat_h, feat_w])
                mask_crop = mask[:, l_miny:l_maxy, l_minx:l_maxx]
                mask_flatten.append(mask_crop.reshape(1, l_h * l_w))

            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            spatial_shapes = torch.stack(spatial_shapes)
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
            )

            all_src.append(src_flatten)
            all_src_padding_mask.append(mask_flatten)
            all_src_spatial_shapes.append(spatial_shapes)
            all_level_start_index.append(level_start_index)

        # NOTE just for now, return only the crop information
        return None, crop_infos

        # obtain edge features
        crop_coords = torch.tensor(crop_coords).cuda(gpu_id)
        crop_enc_a = pos_encode_2d(x=crop_coords[:, 0], y=crop_coords[:, 1])
        crop_enc_b = pos_encode_2d(x=crop_coords[:, 2], y=crop_coords[:, 3])
        crop_enc = torch.cat([crop_enc_a, crop_enc_b], dim=-1)
        crop_feats = edge_input_fc(crop_enc.unsqueeze(0))[0]
        tgt2 = self.with_pos_embed(crop_feats, crop_enc)

        assert len(tgt2) < 64
        num_edges = len(tgt2)
        tgt2 = tgt2.reshape(num_edges, 1, -1)

        reference_points = torch.stack(reference_points).reshape(num_edges, 1, -1)
        reference_points = (
            reference_points[:, :, None] * image_data["valid_ratios"][:, None]
        )

        all_src = torch.cat(all_src, dim=0)
        all_src_padding_mask = torch.cat(all_src_padding_mask, dim=0)

        # cross attention
        tgt2 = self.cross_attn(
            tgt2,
            reference_points,
            all_src,
            all_src_spatial_shapes[0],
            all_level_start_index[0],
            all_src_padding_mask,
        )

        return tgt2.mean(dim=(0, 1)), crop_infos


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
    ):
        super().__init__()
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
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

    # def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
    #             src_padding_mask=None, key_padding_mask=None, get_image_feat=True):
    def forward(
        self, tgt, query_pos, image_feats, key_padding_mask=None, get_image_feat=True
    ):

        # cross attention
        if get_image_feat:
            tgt = tgt + self.dropout1(image_feats)
            tgt = self.norm1(tgt)

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

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoderLayer2(nn.Module):
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
        # cross attention
        self.linear0 = nn.Linear(int(d_model * 2.5), d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
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

    # def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
    #             src_padding_mask=None, key_padding_mask=None, get_image_feat=True):
    def forward(
        self,
        tgt,
        query_pos,
        data,
        image_data,
        key_padding_mask=None,
        get_image_feat=True,
        img_side_len=256,
        interp_dist=128,
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

        if get_image_feat:
            # only do batch size of 1 first
            assert len(tgt) == 1

            # do one forward pass
            tgt2 = self.with_pos_embed(tgt, query_pos)
            new_tgt2 = []
            reference_points = []
            all_src = []
            all_src_spatial_shapes = []
            all_level_start_index = []
            all_src_padding_mask = []
            collect_ind = []

            for edge_i in range(len(tgt[0])):
                (ax, ay, bx, by) = data["edge_coords"][0, edge_i].cpu().numpy()
                edge_shp = LineString([[ax, ay], [bx, by]])
                edge_len = edge_shp.length
                num_interp = np.ceil(edge_len / interp_dist).astype(int)

                for interp_i in range(num_interp + 1):  # +1 to include the end point
                    # determine center of crop
                    edge_center = edge_shp.interpolate(
                        interp_i / num_interp, normalized=True
                    )
                    center_x, center_y = int(edge_center.x), int(edge_center.y)

                    # determine crop boundary
                    (_, img_h, img_w) = data["image_size"][0]
                    crop_minx = max(center_x - img_side_len // 2, 0)
                    crop_maxx = crop_minx + img_side_len
                    crop_miny = max(center_y - img_side_len // 2, 0)
                    crop_maxy = crop_miny + img_side_len

                    if crop_maxx > img_w:
                        crop_maxx = img_w
                        crop_minx = crop_maxx - img_side_len

                    if crop_maxy > img_h:
                        crop_maxy = img_h
                        crop_miny = crop_maxy - img_side_len

                    assert ((crop_maxx - crop_minx) == img_side_len) and (
                        (crop_maxy - crop_miny) == img_side_len
                    )

                    # determine the cropped line's coordinates
                    crop_bbox = box(crop_minx, crop_miny, crop_maxx, crop_maxy)
                    crop_line = crop_bbox.intersection(edge_shp)

                    # the new token feature is the combination of:
                    # 1. Original token feature
                    # 2. The cropped line's coordinates
                    # 3. The edge center coordinate
                    ((crop_ax, crop_ay), (crop_bx, crop_by)) = crop_line.coords

                    pos_x = torch.FloatTensor([crop_ax, crop_bx, center_x]).cuda()
                    pos_y = torch.FloatTensor([crop_ay, crop_by, center_y]).cuda()
                    pos_feats = pos_encode_2d(pos_x, pos_y).flatten()

                    _new_tgt2 = torch.cat([tgt2[0, edge_i], pos_feats], dim=0)
                    new_tgt2.append(_new_tgt2)

                    # but the reference point is in the local coordinate space
                    ref_x = (center_x - crop_minx) / img_side_len
                    ref_y = (center_y - crop_miny) / img_side_len
                    reference_points.append(torch.FloatTensor([ref_x, ref_y]).cuda())

                    # figure out where the new crop would be in each feature level
                    src_flatten = []
                    mask_flatten = []
                    spatial_shapes = []
                    side_lens = [32, 16, 8, 4]

                    for feat_l, (feat_h, feat_w) in enumerate(
                        image_data["spatial_shapes"]
                    ):
                        ratio_h = img_h / feat_h
                        ratio_w = img_w / feat_w

                        l_minx = (crop_minx / ratio_w).round().long()
                        l_miny = (crop_miny / ratio_h).round().long()
                        l_maxx = l_minx + side_lens[feat_l]
                        l_maxy = l_miny + side_lens[feat_l]
                        # l_maxx = (crop_maxx / ratio_w).round().long()
                        # l_maxy = (crop_maxy / ratio_h).round().long()

                        l_h = l_maxy - l_miny
                        l_w = l_maxx - l_minx
                        assert (l_h == l_w) and (l_h in side_lens)
                        spatial_shapes.append(torch.tensor([l_h, l_w]).cuda())

                        start = image_data["level_start_index"][feat_l]
                        end = start + feat_h * feat_w
                        assert end <= image_data["src_flatten"].shape[1]

                        src = image_data["src_flatten"][:, start:end]
                        src = src.reshape([1, feat_h, feat_w, -1])
                        src_crop = src[:, l_miny:l_maxy, l_minx:l_maxx]
                        src_flatten.append(src_crop.reshape(1, l_h * l_w, -1))

                        mask = image_data["mask_flatten"][:, start:end]
                        mask = mask.reshape([1, feat_h, feat_w])
                        mask_crop = mask[:, l_miny:l_maxy, l_minx:l_maxx]
                        mask_flatten.append(mask_crop.reshape(1, l_h * l_w))

                    src_flatten = torch.cat(src_flatten, 1)
                    mask_flatten = torch.cat(mask_flatten, 1)
                    spatial_shapes = torch.stack(spatial_shapes)
                    level_start_index = torch.cat(
                        (
                            spatial_shapes.new_zeros((1,)),
                            spatial_shapes.prod(1).cumsum(0)[:-1],
                        )
                    )

                    all_src.append(src_flatten)
                    all_src_padding_mask.append(mask_flatten)
                    all_src_spatial_shapes.append(spatial_shapes)
                    all_level_start_index.append(level_start_index)
                    collect_ind.append(edge_i)

            # pad to N*64 batch size if necessary
            num_edges = len(new_tgt2)
            pad_num = (np.ceil(num_edges / 64) * 64 - num_edges).astype(int)

            for _ in range(pad_num):
                new_tgt2.append(torch.zeros_like(new_tgt2[0]))
                reference_points.append(torch.zeros_like(reference_points[0]))
                all_src.append(torch.zeros_like(all_src[0]))
                all_src_padding_mask.append(torch.zeros_like(all_src_padding_mask[0]))

            new_tgt2 = torch.stack(new_tgt2).reshape(num_edges + pad_num, 1, -1)
            new_tgt2 = self.linear0(new_tgt2)

            reference_points = torch.stack(reference_points).reshape(
                num_edges + pad_num, 1, -1
            )
            reference_points = (
                reference_points[:, :, None] * image_data["valid_ratios"][:, None]
            )

            all_src = torch.cat(all_src, dim=0)
            all_src_padding_mask = torch.cat(all_src_padding_mask, dim=0)

            # cross attention
            new_tgt2 = self.cross_attn(
                new_tgt2,
                reference_points,
                all_src,
                all_src_spatial_shapes[0],
                all_level_start_index[0],
                all_src_padding_mask,
            )
            new_tgt2 = new_tgt2[:num_edges]  # remove padding

            pooled_tgt2 = torch.zeros_like(tgt)
            collect_ind = torch.as_tensor(collect_ind)
            for edge_i in range(len(tgt[0])):
                pooled_tgt2[0, edge_i] = new_tgt2[collect_ind == edge_i].mean(
                    dim=(0, 1)
                )

            tgt = tgt + self.dropout1(pooled_tgt2)
            tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, return_intermediate=False, with_sa=True
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        assert not return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.with_sa = with_sa
        assert self.with_sa

    def forward(
        self,
        edge_feats,
        geom_feats,
        image_feats,
        key_padding_mask=None,
        get_image_feat=True,
    ):
        output = edge_feats

        for lid, layer in enumerate(self.layers):
            if self.with_sa:
                output = layer(
                    output,
                    geom_feats,
                    image_feats,
                    key_padding_mask=key_padding_mask,
                    get_image_feat=get_image_feat,
                )
            else:
                assert "Should not be here"
                output = layer(output, edge_feats, data, image_data)

        return output


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
