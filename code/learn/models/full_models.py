# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
from models.mlp import MLP
from models.deformable_transformer_full import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, \
    DeformableTransformerDecoder, DeformableTransformerDecoderLayer, DeformableAttnDecoderLayer
from models.ops.modules import MSDeformAttn
from models.corner_models import PositionEmbeddingSine
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import torch.nn.functional as F
from utils.misc import NestedTensor
from utils.nn_utils import pos_encode_2d


class EdgeEnum(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_feature_levels, backbone_strides, backbone_num_channels):
        super(EdgeEnum, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.img_pos = PositionEmbeddingSine(hidden_dim // 2)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
        self.edge_input_fc = nn.Linear(input_dim * 2, hidden_dim)

        self.output_fc = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim // 2, output_dim=2, num_layers=2)

        # self.mlm_embedding = nn.Embedding(3, input_dim)
        self.transformer = EdgeTransformer(d_model=hidden_dim, nhead=8, num_encoder_layers=1,
                                           num_decoder_layers=6, dim_feedforward=1024, dropout=0.1)

        self._reset_parameters()


    def _reset_parameters(self):
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MSDeformAttn):
        #         m._reset_parameters()

        # xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        # constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)


    @staticmethod
    def get_ms_feat(xs, img_mask):
        out: Dict[str, NestedTensor] = {}
        # out = list()
        for name, x in sorted(xs.items()):
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
            # out.append(NestedTensor(x, mask))
        return out


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


    def prepare_image_feat(self, crop, backbone):
        image = torch.tensor(crop['image']).cuda().unsqueeze(0)
        with torch.no_grad():
            image_feats, feat_mask, _ = backbone(image)
        features = self.get_ms_feat(image_feats, feat_mask)

        srcs = []
        masks = []
        all_pos = []

        new_features = list()
        for name, x in sorted(features.items()):
            new_features.append(x)
        features = new_features

        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            mask = mask.to(src.device)
            srcs.append(self.input_proj[l](src))
            pos = self.img_pos(src).to(src.dtype)
            all_pos.append(pos)
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = feat_mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0].to(src.device)
                pos_l = self.img_pos(src).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                all_pos.append(pos_l)

        # prepare input for encoder
        pos_embeds = all_pos

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        image_data = {
            'src_flatten': src_flatten,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'valid_ratios': valid_ratios,
            'lvl_pos_embed_flatten': lvl_pos_embed_flatten,
            'mask_flatten': mask_flatten
        }
        return image_data


    def forward(self, data, backbone):
        # encode edges
        edge_coords = data['edge']
        edge_feats_a = pos_encode_2d(x=edge_coords[:,0], y=edge_coords[:,1])
        edge_feats_b = pos_encode_2d(x=edge_coords[:,2], y=edge_coords[:,3])
        edge_feats = torch.cat([edge_feats_a, edge_feats_b], dim=-1)
        edge_feats = self.edge_input_fc(edge_feats)

        # prepare image features
        for batch_crops in data['crops']:
            for crop in batch_crops:
                crop['image_data'] = self.prepare_image_feat(crop, backbone)

        # edge_center = (edge_coords[:, :, 0, :].float() + edge_coords[:, :, 1, :].float()) / 2
        # edge_center = edge_center / feat_mask.shape[1]

        # logits_per_edge, logits_hb, logits_rel, selection_ids, s2_attn_mask, s2_gt_values = self.transformer(srcs,
        #                                                                                                      masks,
        #                                                                                                      all_pos,
        #                                                                                                      edge_inputs,
        #                                                                                                      edge_center,
        #                                                                                                      gt_values,
        #                                                                                                      edge_masks,
        #                                                                                                      corner_nums,
        #                                                                                                      max_candidates,
        #                                                                                                      do_inference)

        self.transformer(edge_feats, data)

        return logits_per_edge, logits_hb, logits_rel, selection_ids, s2_attn_mask, s2_gt_values


class EdgeTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 ):
        super(EdgeTransformer, self).__init__()

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_attn_layer = DeformableAttnDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points)
        self.per_edge_decoder = DeformableTransformerDecoder(decoder_attn_layer, 1, False, with_sa=False)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.relational_decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                                               return_intermediate_dec, with_sa=True)

        # self.reference_points = nn.Linear(d_model, 2)

        self.gt_label_embed = nn.Embedding(3, d_model)

        self.input_fc_hb = MLP(input_dim=2 * d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)
        self.input_fc_rel = MLP(input_dim=2 * d_model, hidden_dim=d_model, output_dim=d_model, num_layers=2)

        self.output_fc_1 = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2)
        self.output_fc_2 = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2)
        self.output_fc_3 = MLP(input_dim=d_model, hidden_dim=d_model // 2, output_dim=2, num_layers=2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, edge_feats, data):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []

        for batch_crops in data['crops']:
            for crop in batch_crops:
                image_data = crop['image_data']
                with torch.no_grad():
                    memory = self.encoder(image_data['src_flatten'],
                                          image_data['spatial_shapes'],
                                          image_data['level_start_index'],
                                          image_data['valid_ratios'],
                                          image_data['lvl_pos_embed_flatten'],
                                          image_data['mask_flatten'])
                image_data['memory'] = memory

        hs_per_edge, _ = self.per_edge_decoder(edge_feats, data)

        logits_per_edge = self.output_fc_1(hs_per_edge).permute(0, 2, 1)

        # relational decoder with image feature
        gt_info = self.gt_label_embed(filtered_gt_values)
        hybrid_prim_hs = self.input_fc_hb(torch.cat([filtered_hs, gt_info], dim=-1))

        hs, inter_references = self.relational_decoder(hybrid_prim_hs, filtered_rp, memory,
                                                       spatial_shapes, level_start_index, valid_ratios, filtered_query,
                                                       mask_flatten,
                                                       key_padding_mask=filtered_mask, get_image_feat=True)

        logits_final_hb = self.output_fc_2(hs).permute(0, 2, 1)

        # relational decoder without image feature
        rel_prim_hs = self.input_fc_rel(torch.cat([filtered_query, gt_info], dim=-1))

        hs_rel, _ = self.relational_decoder(rel_prim_hs, filtered_rp, memory,
                                            spatial_shapes, level_start_index, valid_ratios, filtered_query,
                                            mask_flatten,
                                            key_padding_mask=filtered_mask, get_image_feat=False)

        logits_final_rel = self.output_fc_3(hs_rel).permute(0, 2, 1)

        return logits_per_edge, logits_final_hb, logits_final_rel, selected_ids, filtered_mask, filtered_gt_values