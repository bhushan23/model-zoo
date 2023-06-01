from collections import OrderedDict

import ane_transformers
import clip
import torch
import torch.nn as nn
from ane_transformers.reference.multihead_attention import MultiHeadAttention

EPS = 1e-5


class LayerNormANE(ane_transformers.reference.layer_norm.LayerNormANE):
    """
    Over-ride from ane_transformers due to formulae mismatch between
    torch and ane_transformers.
    """

    def forward(self, inputs: torch.Tensor):
        # Overrides only the forward for LayerNormANE
        # since the formula mismatch exists here.
        input_rank = len(inputs.size())
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())

        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out = zero_mean * denom

        # Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
        # apply scale and bias terms in opposite orders. In order to accurately restore a
        # state_dict trained using the former into the the latter, we adjust the bias term
        if self.elementwise_affine:
            out = out * self.weight.view(1, self.num_channels, 1, 1) + self.bias.view(
                1, self.num_channels, 1, 1
            )
        return out


class ResidualAttentionBlock(clip.model.ResidualAttentionBlock):
    """
    Overide Residual Attention module from Clip to replace its modules with ANE compliant modules.
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__(d_model, n_head, attn_mask)
        setattr(
            self,
            "attn",
            MultiHeadAttention(embed_dim=d_model, n_head=n_head, dropout=0.0),
        )
        setattr(self, "ln_1", LayerNormANE(d_model, eps=EPS))
        setattr(self, "ln_2", LayerNormANE(d_model, eps=EPS))
        setattr(
            self,
            "mlp",
            nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Conv2d(d_model, d_model * 4, 1)),
                        ("gelu", clip.model.QuickGELU()),
                        ("c_proj", nn.Conv2d(d_model * 4, d_model, 1)),
                    ]
                )
            ),
        )

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(q=x, k=x, v=x, qk_mask=self.attn_mask)[0]


class Transformer(clip.model.Transformer):
    """
    Overide Transformer module from Clip to replace its modules with ANE compliant modules
    and transform the weights per the new modules.
    """

    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__(width, layers, heads, attn_mask)
        setattr(
            self,
            "resblocks",
            nn.Sequential(
                *[
                    ResidualAttentionBlock(width, heads, attn_mask)
                    for _ in range(layers)
                ]
            ),
        )

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)
        self._register_load_state_dict_pre_hook(qkv_proj)


class VisionTransformer(clip.model.VisionTransformer):
    """
    Overide Vision Transformer module from Clip to replace its modules with ANE compliant modules.
    """

    @classmethod
    def from_clip(cls, clip_model: nn.Module) -> nn.Module:
        vision_transformer = clip_model.visual
        input_resolution = vision_transformer.input_resolution
        patch_size = vision_transformer.conv1.kernel_size[0]
        width = vision_transformer.transformer.width
        layers = vision_transformer.transformer.layers

        heads = width // 64
        output_dim = vision_transformer.output_dim

        ane_vision_transformer = cls(
            input_resolution, patch_size, width, layers, heads, output_dim
        )
        ane_vision_transformer.load_state_dict(vision_transformer.state_dict())
        return ane_vision_transformer

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        setattr(self, "ln_pre", LayerNormANE(width, eps=EPS))
        setattr(self, "ln_post", LayerNormANE(width, eps=EPS))
        setattr(self, "transformer", Transformer(width, layers, heads))
        setattr(self, "proj_conv", nn.Conv2d(width, output_dim, 1))

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)
        self._register_load_state_dict_pre_hook(positional_embedding_squeeze)
        self._register_load_state_dict_pre_hook(proj_conv_weights)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        # shape = [batch, width, grid, grid]
        batch, width, grid, _ = x.shape
        x = x.reshape(batch, width, 1, grid**2)  # shape = [batch, width, grid ** 2]

        class_embedding = self.class_embedding.unsqueeze(-1).unsqueeze(-1)
        x = torch.cat(
            [
                torch.zeros(batch, width, 1, 1) + class_embedding,
                x,
            ],
            dim=-1,
        )

        positional_embedding_t = (
            self.positional_embedding.transpose(1, 0).unsqueeze(-2).unsqueeze(0)
        )
        x = x + positional_embedding_t
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, :, :, 0:1])

        if self.proj_conv is not None:
            x = self.proj_conv(x)

        return x


def qkv_proj(
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
):
    """
    For Clip's transformer module, qkv are concatenated and stored 'in_proj' key in state_dict
    and ANE Transformer's transformer module, requires the three to be split up and loaded.
    The layout for self attenion is: q, k and then v.
    TODO: For encoder attention q==k != v there's a diff layout (not implemented here).
    """

    keys = state_dict.copy().keys()
    keys_to_remove = set()
    for k in keys:
        if "in_proj" in k:
            for index, val in enumerate(["q", "k", "v"]):
                keys_to_remove.add(k)
                if "in_proj_weight" in k:
                    # [q_weight | k_weight | v_weight] with shape: [3 * proj_dim, width]
                    new_key = k.replace("in_proj_weight", f"{val}_proj.weight")
                    e_dim = state_dict[k].shape[1]
                    state_dict[new_key] = state_dict[k][
                        index * e_dim : (index + 1) * e_dim, :, :, :
                    ]
                else:
                    # [q_bias | k_bias | v_bias] with shape: [3 * proj_dim]
                    new_key = k.replace("in_proj_bias", f"{val}_proj.bias")
                    e_dim = state_dict[k].shape[0] // 3
                    state_dict[new_key] = state_dict[k][
                        index * e_dim : (index + 1) * e_dim
                    ]

    for k in keys_to_remove:
        del state_dict[k]


def linear_to_conv2d_map(
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
):
    """
    Transform the linear layer weights to coonvolution weights
    so that the ransformed module is numerically correct.
    """
    for k in state_dict:
        is_internal_proj = all(substr in k for substr in ["lin", ".weight"])
        is_output_proj = all(substr in k for substr in ["classifier", ".weight"])
        if is_internal_proj or is_output_proj:
            if len(state_dict[k].shape) == 2:
                state_dict[k] = state_dict[k][:, :, None, None]

        if state_dict[k].dim() == 2:
            state_dict[k] = state_dict[k].unsqueeze(-1).unsqueeze(-1)


def positional_embedding_squeeze(
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
):
    """
    The positional embeddings must be mapped correctly. Needs to
    exapand dims at the end to match layout of the module.

    """
    for k in state_dict:
        if state_dict[k].dim() == 4 and k in ["positional_embedding", "proj"]:
            state_dict[k] = state_dict[k].squeeze(-1).squeeze(-1)


def proj_conv_weights(
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
):
    """Weights and bias for Conv1D is converted for Conv2D."""
    proj_key = "proj"
    proj_weights = state_dict[proj_key]
    width, output_dim = proj_weights.shape

    # Transpose and unsqueeze weights. Set bias to zero
    proj_weights = proj_weights.transpose(0, 1)
    proj_weights = proj_weights[:, :, None, None]
    state_dict["proj_conv.weight"] = proj_weights
    state_dict["proj_conv.bias"] = torch.zeros(output_dim)
