# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Encoder-decoder transformer layers for self/cross attention."""

from copy import deepcopy
import einops
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .eg_attention import MultiheadAttention

def calc_pairwise_locs(obj_centers, point_center, eps=1e-10):
    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
                    - einops.repeat(point_center, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs ** 2, 3) + eps)  # (b, l, l)
    norm_pairwise_dists = pairwise_dists

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2] ** 2, 3) + eps)

    pairwise_locs = torch.stack(
        [norm_pairwise_dists, pairwise_locs[..., 2] / pairwise_dists,
         pairwise_dists_2d / pairwise_dists, pairwise_locs[..., 1] / pairwise_dists_2d,
         pairwise_locs[..., 0] / pairwise_dists_2d],
        dim=3
    )
    return pairwise_locs

class GMA(nn.Module):
    def __init__(
        self, d_model, n_head, dropout=0.1, spatial_dim=5, threshold=0.5,
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' %(d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_dim = spatial_dim

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.threshold = threshold

        self.n_head = n_head

        self.lang_cond_fc = nn.Linear(d_model, d_model//4)
        self.pairwise_loc_fc = nn.Sequential(
            nn.Linear(5, (d_model//self.n_head)//4),
            nn.ReLU(),
            nn.Linear((d_model//self.n_head)//4, (d_model//self.n_head)//4),
        )

    def hard_sigmoid(self, logits):
        y_soft = logits.sigmoid()

        indices = (y_soft > self.threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1], indices[2], indices[3]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft

        return ret

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None, txt_embeds=None):
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        spatial_weights = self.lang_cond_fc(txt_embeds.unsqueeze(1))
        spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head)
        spatial_weights = einops.repeat(spatial_weights, 'h b 1 d -> h b q d', q=256)
        if self.spatial_n_head == 1:
            spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
        pairwise_locs = self.pairwise_loc_fc(pairwise_locs)
        loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs)# + spatial_bias
        loc_attn = loc_attn / np.sqrt(q.shape[-1])
        loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)


        fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
        fused_attn = torch.softmax(fused_attn, 3)
        
        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        return output, fused_attn


class PECA(nn.Module):
    def __init__(
            self, d_model, n_head, dropout=0.1, spatial_dim=5,
            spatial_attn_fusion='cond', threshold=0.5,
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' % (d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.gp_linear = nn.Linear(d_model, d_model)
        self.kp_linear = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, q_p, p_feat, key_padding_mask=None):

        q_p = einops.rearrange(self.gp_linear(q_p), 'b t (head k) -> head b t k', head=self.n_head)
        k_p = einops.rearrange(k, 'b t (head k) -> head b t k', head=self.n_head)
        p_feat = einops.rearrange(p_feat, 'b l (head k) -> head b l k', head=self.n_head)
        k_p = torch.einsum('hblk,hbtk->hblt', k_p, p_feat) / np.sqrt(p_feat.shape[-1])
        k_p = torch.softmax(k_p, dim=-1)
        k_p = torch.einsum('hblt,hbtv->hblv', k_p, q_p)

        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        k = k + k_p
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)

        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)

        attn = torch.softmax(attn, dim=-1)
        output = torch.einsum('hblt,hbtv->hblv', attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.fc(output)

        return output, attn

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

# BRIEF Position Embedding
class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

# BRIEF Cross-attention between language and vision
class CrossAttentionLayer(nn.Module):
    """Cross-attention between language and vision."""

    def __init__(self, d_model=256, dropout=0.1, n_heads=8,
                 dim_feedforward=256, use_butd_enc_attn=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.use_butd_enc_attn = use_butd_enc_attn

        # Cross attention from lang to vision
        self.cross_lv = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_lv = nn.Dropout(dropout)
        self.norm_lv = nn.LayerNorm(d_model)
        self.ffn_lv = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm_lv2 = nn.LayerNorm(d_model)

        # Cross attention from vision to lang
        self.cross_vl = deepcopy(self.cross_lv)
        self.dropout_vl = nn.Dropout(dropout)
        self.norm_vl = nn.LayerNorm(d_model)
        self.ffn_vl = deepcopy(self.ffn_lv)
        self.norm_vl2 = nn.LayerNorm(d_model)

        if use_butd_enc_attn:
            self.cross_d = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout
            )
            self.dropout_d = nn.Dropout(dropout)
            self.norm_d = nn.LayerNorm(d_model)

    def forward(self, vis_feats, vis_key_padding_mask, text_feats,
                text_key_padding_mask, pos_feats,
                detected_feats=None, detected_mask=None):
        """Forward pass, vis/pos_feats (B, V, F), lang_feats (B, L, F)."""
        # produce key, query, value for image
        qv = kv = vv = vis_feats
        qv = qv + pos_feats  # add pos. feats only on 【query】

        # produce key, query, value for text
        qt = kt = vt = text_feats

        # step cross attend language to vision
        text_feats2 = self.cross_lv(
            query=qt.transpose(0, 1),
            key=kv.transpose(0, 1),
            value=vv.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=vis_key_padding_mask  # (B, V)
        )[0].transpose(0, 1)
        text_feats = text_feats + self.dropout_lv(text_feats2)
        text_feats = self.norm_lv(text_feats)
        text_feats = self.norm_lv2(text_feats + self.ffn_lv(text_feats))

        # step cross attend vision to language
        vis_feats2 = self.cross_vl(
            query=qv.transpose(0, 1),
            key=kt.transpose(0, 1),
            value=vt.transpose(0, 1),
            attn_mask=None,
            key_padding_mask=text_key_padding_mask  # (B, L)
        )[0].transpose(0, 1)
        vis_feats = vis_feats + self.dropout_vl(vis_feats2)
        vis_feats = self.norm_vl(vis_feats)

        # step cross attend vision to boxes
        if detected_feats is not None and self.use_butd_enc_attn:
            vis_feats2 = self.cross_d(
                query=vis_feats.transpose(0, 1),
                key=detected_feats.transpose(0, 1),
                value=detected_feats.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=detected_mask
            )[0].transpose(0, 1)
            vis_feats = vis_feats + self.dropout_d(vis_feats2)
            vis_feats = self.norm_d(vis_feats)

        # FFN
        vis_feats = self.norm_vl2(vis_feats + self.ffn_vl(vis_feats))

        return vis_feats, text_feats

# BRIEF text self-attention
class TransformerEncoderLayerNoFFN(nn.Module):
    """TransformerEncoderLayer but without FFN."""

    def __init__(self, d_model, nhead, dropout):
        """Intialize same as Transformer (without FFN params)."""
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer (same as parent class).

        Args:
            src: (S, B, F)
            src_mask: the mask for the src sequence (optional)
            src_key_padding_mask: (B, S) mask for src keys per batch (optional)
        Shape:
            see the docs in Transformer class.
        Return_shape: (S, B, F)
        """
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

# BRIEF vision self-attention
class PosTransformerEncoderLayerNoFFN(TransformerEncoderLayerNoFFN):
    """TransformerEncoderLayerNoFFN but additionaly add pos_embed in query."""

    def __init__(self, d_model, nhead, dropout):
        """Intialize same as parent class."""
        super().__init__(d_model, nhead, dropout)

    def forward(self, src, pos, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer (same as parent class).

        Args:
            src: (S, B, F)  
            pos: (S, B, F) positional embeddings
            src_mask: the mask for the src sequence (optional)
            src_key_padding_mask: (B, S) mask for src keys per batch (optional)
        Shape:
            see the docs in Transformer class.
        Return_shape: (S, B, F)
        """
        src2 = self.self_attn(
            src + pos, src + pos, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src

# BRIEF vision text self attention and cross attention
class BiEncoderLayer(nn.Module):
    """Self->cross layer for both modalities."""

    def __init__(self, d_model=256, dropout=0.1, activation="relu", n_heads=8,
                 dim_feedforward=256,
                 self_attend_lang=True, self_attend_vis=True,
                 use_butd_enc_attn=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        # self attention in language
        if self_attend_lang:
            self.self_attention_lang = TransformerEncoderLayerNoFFN(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout
            )
        else:
            self.self_attention_lang = None

        # self attention in vision
        if self_attend_vis:
            self.self_attention_visual = PosTransformerEncoderLayerNoFFN(
                d_model=d_model,
                nhead=n_heads,
                dropout=dropout
            )
        else:
            self.self_attention_visual = None

        # cross attention in language and vision
        self.cross_layer = CrossAttentionLayer(
            d_model, dropout, n_heads, dim_feedforward,
            use_butd_enc_attn
        )
    
    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points={}, detected_feats=None,
                detected_mask=None,spatial_point_xyz=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        # STEP 1. Self attention for vision
        if self.self_attention_visual is not None:
            vis_feats = self.self_attention_visual(
                vis_feats.transpose(0, 1),
                pos_feats.transpose(0, 1),
                src_key_padding_mask=padding_mask
            ).transpose(0, 1)

        # STEP 2. Self attention for language
        if self.self_attention_lang is not None:
            text_feats = self.self_attention_lang(
                text_feats.transpose(0, 1),
                src_key_padding_mask=text_padding_mask
            ).transpose(0, 1)

        # STEP 3. Cross attention
        vis_feats, text_feats = self.cross_layer(
            vis_feats=vis_feats,
            vis_key_padding_mask=padding_mask,
            text_feats=text_feats,
            text_key_padding_mask=text_padding_mask,
            pos_feats=pos_feats,
            detected_feats=detected_feats,
            detected_mask=detected_mask
        )

        return vis_feats, text_feats


# BRIEF 
class BiEncoder(nn.Module):
    """Encode jointly language and vision."""

    def __init__(self, bi_layer, num_layers):
        """Pass initialized BiEncoderLayer and number of such layers."""
        super().__init__()
        self.layers = _get_clones(bi_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, vis_feats, pos_feats, padding_mask, text_feats,
                text_padding_mask, end_points={},
                detected_feats=None, detected_mask=None,spatial_point_xyz=None):
        """Forward pass, feats (B, N, F), masks (B, N), diff N for V/L."""
        for i, layer in enumerate(self.layers):
            vis_feats, text_feats = layer(
                vis_feats,
                pos_feats,
                padding_mask,
                text_feats,
                text_padding_mask,
                end_points,
                detected_feats=detected_feats,
                detected_mask=detected_mask,
                spatial_point_xyz=spatial_point_xyz
            )
            if 'lv_attention' in end_points:
                end_points['lv_attention%d' % i] = end_points['lv_attention']
        return vis_feats, text_feats


class SWA(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.2):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """

        query = self.with_pos_embed(query, pe)
        B = query.shape[1]
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B * self.nhead, query.shape[0],
                                                                                k.shape[0])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=None,
                                                          attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v)
        output = self.dropout(output)
        output = output + query
        output = self.norm(output)

        return output.transpose(0, 1), output_weight, src_weight  # (b, n_q, d_model), (b, n_q, n_v)


# BRIEF Transformer decoder
class BiDecoderLayer(nn.Module):
    """Self->cross_l->cross_v layer for proposals."""

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 self_position_embedding='loc_learned', butd=False, super_refine=False):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()

        # STEP 1. Self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # STEP 2. Cross attention in language
        # self.cross_l = nn.MultiheadAttention(
        #     d_model, n_heads, dropout=dropout
        # )
        self.cross_l = PECA(
                d_model=d_model,n_head=n_heads,
                dropout=dropout
        )
        self.dropout_l = nn.Dropout(dropout)
        self.norm_l = nn.LayerNorm(d_model)

        if butd:
            # STEP 3. Cross attention in detected boxes
            self.cross_d = deepcopy(self.self_attn)
            self.dropout_d = nn.Dropout(dropout)
            self.norm_d = nn.LayerNorm(d_model)

        # STEP 4. Cross attention in vision
        self.cross_v = GMA(
            d_model=d_model,n_head=n_heads,
            dropout=dropout
        )
        self.dropout_v = nn.Dropout(dropout)
        self.norm_v = nn.LayerNorm(d_model)

        # STEP 4-1. Cross attention in superpoints
        self.cross_s = SWA(
            d_model, n_heads, dropout
        )

        # STEP 4-2. Add point query and super query
        self.norm_sp = nn.LayerNorm(d_model)

        # STEP 5. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # STEP 6. superpoint refinement
        if super_refine:
            self.refine_super = nn.MultiheadAttention(
                d_model, n_heads,
                dropout=dropout
            )
            self.refine_norm = nn.LayerNorm(d_model)
            self.refine_dropout1 = nn.Dropout(dropout)

            self.refine_super_text = deepcopy(self.refine_super)
            self.refine_norm_text = nn.LayerNorm(d_model)
            self.refine_dropout_text = nn.Dropout(dropout)

            self.refine_super_ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            self.refine_norm2 = nn.LayerNorm(d_model)
        else:
            self.refine_super = None

        # STEP 6. Positional embeddings
        if self_position_embedding == 'xyz_learned':
            self.self_posembed = PositionEmbeddingLearned(3, d_model)
        elif self_position_embedding == 'loc_learned':
            self.self_posembed = PositionEmbeddingLearned(6, d_model)
        else:
            self.self_posembed = None

    def forward(self, query, vis_feats, lang_feats, query_pos,
                padding_mask, text_key_padding_mask,
                detected_feats=None, detected_mask=None,
                xyz_embed=None, points_xyz=None, super_features=None, attn_mask_list=None):
        """
        Forward pass.
        Args:
            query: (B, N, F)        ([B, N=256, 288])
            vis_feats: (B, V, F)    ([B, 1024, 288])
            lang_feats: (B, L, F)   ([B, L, 288])
            query_pos: (B, N, 3or6) ([B, N=256, 6])
            padding_mask: (B, N) (for query)
            text_key_padding_mask: (B, L)   ([B, L])
        Returns:
            query: (B, N, F)
        """
        # step 0. query_pos position embedding, 6-->288
        if self.self_posembed is not None:
            query_xyz = query_pos[:, :, :3]
            query_pos = self.self_posembed(query_pos)
            query_pos = query_pos.transpose(1, 2).contiguous()
        else:
            query_pos = torch.zeros_like(query, device=query.device)
        query = query.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)

        # step 1. self-attention
        query2 = self.self_attn(
            query + query_pos, query + query_pos, query,
            attn_mask=None,
            key_padding_mask=padding_mask
        )[0]
        query = self.norm1(query + self.dropout1(query2))
                    
        # step 2. Cross attend to language
        query2 = self.cross_l(
            q=(query + query_pos).transpose(0, 1),
            k=lang_feats,
            v=lang_feats,
            q_p=xyz_embed.transpose(1, 2),
            p_feat=vis_feats,
            key_padding_mask=text_key_padding_mask  # (B, L)
        )[0]
        query2 = query2.transpose(0, 1).contiguous()
        query = self.norm_l(query + self.dropout_l(query2))

        # step 3. Cross attend to detected boxes
        if detected_feats is not None:
            query2 = self.cross_d(
                query=query + query_pos,
                key=detected_feats.transpose(0, 1),
                value=detected_feats.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=detected_mask
            )[0]
            query = self.norm_d(query + self.dropout_d(query2))

        spatial_feature = calc_pairwise_locs(query_xyz, points_xyz)

        # step 4. Cross attend to vision
        query2 = self.cross_v(
            q=(query + query_pos).transpose(0, 1),
            k=vis_feats,
            v=vis_feats,
            pairwise_locs=spatial_feature,
            key_padding_mask=padding_mask,
            txt_embeds=torch.max(lang_feats,dim=1)[0]
        )[0]
        query2 = query2.transpose(0, 1).contiguous()
        query_v = self.norm_v(query + self.dropout_v(query2))

        query_s_list = []
        for bs in range(query.shape[1]):
            bs_query = query[:, bs].unsqueeze(1)
            bs_query, _, src_weight = self.cross_s(
                super_features[bs].unsqueeze(0).transpose(1, 2).transpose(0, 1), bs_query,
                attn_mask=attn_mask_list[bs])
            query_s_list.append(bs_query)
        query_s = torch.cat(query_s_list)

        query = self.norm_sp(query_v + query_s.transpose(0, 1))

        # step 5. FFN + layer norm
        query = self.norm2(query + self.ffn(query))

        # step 6. Refine superpoint feature
        if self.refine_super is not None:
            super_features_list = []
            for bs in range(query.shape[1]):
                bs_query = query[:, bs].unsqueeze(1)
                super_feat = super_features[bs].transpose(0, 1).unsqueeze(1).contiguous()
                refine_super_feat = self.refine_super(
                    super_feat, bs_query, bs_query,
                    attn_mask=None,
                    key_padding_mask=None
                )[0]
                refine_super_feat = self.refine_norm(super_feat + self.refine_dropout1(refine_super_feat))

                bs_lang_feat = lang_feats[bs].unsqueeze(1)
                bs_text_key_padding_mask = text_key_padding_mask[bs].unsqueeze(0)
                refine_super_feat2 = self.refine_super_text(
                    refine_super_feat, bs_lang_feat, bs_lang_feat,
                    attn_mask=None,
                    key_padding_mask=bs_text_key_padding_mask
                )[0]
                refine_super_feat = self.refine_norm_text(refine_super_feat + self.refine_dropout_text(refine_super_feat2))

                refine_super_feat = self.refine_norm2(refine_super_feat + self.refine_super_ffn(refine_super_feat))
                super_features_list.append(refine_super_feat.squeeze(1).transpose(0, 1).contiguous())
            super_features = super_features_list

        return query.transpose(0, 1).contiguous(), super_features
