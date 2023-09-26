# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# --------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_


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


class LocalAttention(nn.Module):
    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):

        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):

        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]

        x_total = einops.rearrange(
            x,
            "b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c",
            h1=self.window_size[0],
            w1=self.window_size[1],
        )  # B x Nr x Ws x C

        x_total = einops.rearrange(x_total, "b m n c -> (b m) n c")

        qkv = self.proj_qkv(x_total)  # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [
            einops.rearrange(t, "b n (h c1) -> b h n c1", h=self.heads)
            for t in [q, k, v]
        ]
        attn = torch.einsum("b h m c, b h n c -> b h m n", q, k)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn : (b * nW) h w w
            # mask : nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(
                attn, "(b n) h w1 w2 -> b n h w1 w2", n=nW, h=self.heads, w1=ww, w2=ww
            ) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, "b n h w1 w2 -> (b n) h w1 w2")
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum("b h m n, b h n c -> b h m c", attn, v)
        x = einops.rearrange(x, "b h n c1 -> b n (h c1)")
        x = self.proj_drop(self.proj_out(x))  # B' x N x C
        x = einops.rearrange(
            x,
            "(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)",
            r1=r1,
            r2=r2,
            h1=self.window_size[0],
            w1=self.window_size[1],
        )  # B x C x H x W

        return x, None, None


class ShiftWindowAttention(LocalAttention):
    def __init__(
        self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size
    ):

        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)  # H W
        h_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size[1]),
            slice(-self.window_size[1], -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(
            img_mask,
            "(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)",
            h1=self.window_size[0],
            w1=self.window_size[1],
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        shifted_x = torch.roll(
            x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3)
        )
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None


class DAttentionBaseline(nn.Module):
    def __init__(
        self,
        q_size,
        kv_size,
        n_heads,
        n_head_channels,
        n_groups,
        attn_drop,
        proj_drop,
        stride,
        offset_range_factor,
        use_pe,
        dwc_pe,
        no_off,
        fixed_pe,
        stage_idx,
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels**-0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        # self.conv_offset = nn.Sequential(
        #     nn.Conv2d(
        #         self.n_group_channels,
        #         self.n_group_channels,
        #         kk,
        #         stride,
        #         kk // 2,
        #         groups=self.n_group_channels,
        #     ),
        #     LayerNormProxy(self.n_group_channels),
        #     nn.GELU(),
        #     nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
        # )
        self.linear_offset = nn.Linear(384, 2, bias=False)

        # self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_q = nn.Linear(self.n_head_channels, self.n_head_channels)

        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)

        # self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.n_head_channels, self.n_head_channels)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc
                )
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(
                        self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w
                    )
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(
        self,
        edge_feats,
        edge_coords,
        edge_masks,
        lvls_feats,
        lvls_spatial_shapes,
        lvls_start_index,
        interp_dist=20,
    ):
        edge_coords = edge_coords.float()

        B, N, _ = edge_coords.shape
        assert B == 1
        device = edge_coords.device
        dtype = edge_coords.dtype

        # sample points evently along line
        _edge_coords = edge_coords.reshape(B * N, 4)
        all_samples = []
        for (x0, y0, x1, y1) in _edge_coords.cpu().numpy():
            line_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            num_samples = int(line_len // interp_dist) + 1
            xx = np.linspace(x0, x1, num_samples, endpoint=False)
            yy = np.linspace(y0, y1, num_samples, endpoint=False)
            all_samples.append(np.stack([xx, yy], axis=1))

        # zero-pad jagged sample array
        max_len = max([len(x) for x in all_samples])
        sample_pad_mask = np.zeros((N, max_len), dtype=bool)
        for i in range(len(all_samples)):
            edge_samples = all_samples[i]
            padded_samples = np.pad(
                edge_samples, [[0, max_len - len(edge_samples)], [0, 0]]
            )
            sample_pad_mask[i, len(edge_samples) :] = True
            all_samples[i] = padded_samples

        # [N, L, 2], L = max_len
        all_samples = torch.tensor(all_samples, device=device, dtype=dtype)

        # query nodes
        q = self.proj_q(edge_feats)

        # predict offsets
        # _reference = all_samples.reshape(-1, 2)
        # ref_embed = pos_encode_2d(_reference[:, 0], _reference[:, 1])
        # ref_embed = ref_embed.reshape(B, N, num_samples, 128)

        # q_off = q.clone().unsqueeze(2).repeat(1, 1, num_samples, 1)
        # q_off = torch.cat([q_off, ref_embed], axis=-1)
        # offset = self.linear_offset(q_off).permute(0, 3, 1, 2)
        # Hk, Wk = offset.size(2), offset.size(3)
        # n_sample = Hk * Wk

        # if self.offset_range_factor > 0:
        #     offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(
        #         1, 2, 1, 1
        #     )
        #     offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # offset = einops.rearrange(offset, "b p h w -> b h w p")

        # if self.no_off:
        #     offset = offset.fill(0.0)

        # if self.offset_range_factor >= 0:
        #     pos = offset + all_samples
        # else:
        #     pos = (offset + all_samples).tanh()

        # NOTE only use one resolution map for now
        lvl_start, lvl_end = lvls_start_index[0:2]
        (lvl_h, lvl_w) = lvls_spatial_shapes[0].cpu().numpy().tolist()
        x = lvls_feats[:, lvl_start:lvl_end, :].reshape([B, lvl_h, lvl_w, -1])
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        B, C, H, W = x.shape

        x_sampled = F.grid_sample(
            input=x,
            grid=pos,
            mode="bilinear",
            align_corners=True,
        )  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, N)
        k = self.proj_k(x_sampled).reshape(
            B * self.n_heads, self.n_head_channels, n_sample
        )
        v = self.proj_v(x_sampled).reshape(
            B * self.n_heads, self.n_head_channels, n_sample
        )

        attn = torch.einsum("b c m, b c n -> b m n", q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe:

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(
                    B * self.n_heads, self.n_head_channels, H * W
                )
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_ref_points(H, W, B, dtype, device)

                displacement = (
                    q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2)
                    - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)
                ).mul(0.5)

                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(
                        B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1
                    ),
                    grid=displacement[..., (1, 0)],
                    mode="bilinear",
                    align_corners=True,
                )  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)

                attn = attn + attn_bias

        # ignore padded edges
        pad_masks = torch.repeat_interleave(edge_masks, repeats=num_samples, dim=-1)
        pad_masks = pad_masks.unsqueeze(1).repeat(1, N, 1)
        attn[pad_masks] = float("-inf")

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum("b m n, b c n -> b c m", attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, N)
        out = out.permute(0, 2, 1)  # b c m -> b m c

        y = self.proj_drop(self.proj_out(out))

        return y, pos, all_samples


class TransformerMLP(nn.Module):
    def __init__(self, channels, expansion, drop):

        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module("linear1", nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module("act", nn.GELU())
        self.chunk.add_module("drop1", nn.Dropout(drop, inplace=True))
        self.chunk.add_module("linear2", nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module("drop2", nn.Dropout(drop, inplace=True))

    def forward(self, x):

        _, _, H, W = x.size()
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.chunk(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return einops.rearrange(x, "b h w c -> b c h w")


class TransformerMLPWithConv(nn.Module):
    def __init__(self, channels, expansion, drop):

        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):

        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))

        return x
