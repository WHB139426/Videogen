# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import json
import pdb

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.loaders import PeftAdapterMixin

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models_expand.unet_blocks import (
    CrossAttnDownBlock,
    CrossAttnUpBlock,
    DownBlock,
    UNetMidBlockCrossAttn,
    UpBlock,
    get_down_block,
    get_up_block,
)
from models_expand.resnet import InflatedConv3d, InflatedGroupNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

unet_additional_kwargs = {
    'use_inflated_groupnorm': True, 
    'use_motion_module': True, 
    'motion_module_resolutions': [1, 2, 4, 8], 
    'motion_module_mid_block': False, 
    'motion_module_type': 'Vanilla', 
    'motion_module_kwargs': {
        'num_attention_heads': 8, 
        'num_transformer_block': 1, 
        'attention_block_types': ['Temporal_Self', 'Temporal_Self'], 
        'temporal_position_encoding': True, 
        'temporal_attention_dim_div': 1, 
        'zero_initialize': True
        }
    }

# x = torch.randn(2, 4, 64, 64)
# cond = torch.randn(2, 4, 64, 64)
# new_x = torch.cat([x, cond], dim=1)

# conv_1 = nn.Conv2d(4, 320, kernel_size=3, padding=(1, 1))
# conv_2 = nn.Conv2d(8, 320, kernel_size=3, padding=(1, 1))

# conv_2.weight.data[:,:4,:,:] = conv_1.weight.data
# conv_2.weight.data[:, 4:, :, :].zero_()
# conv_2.bias.data = conv_1.bias.data

# print(conv_1.weight.shape, conv_1.bias.shape)
# print(conv_2.weight.shape, conv_2.bias.shape)
# y1 = conv_1(x)
# y2 = conv_2(new_x)
# print(y1.shape)
# print(y2.shape)
# print(torch.equal(y1, y2))

@dataclass
class UNetConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNetConditionModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,      
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock",
            "CrossAttnDownBlock",
            "CrossAttnDownBlock",
            "DownBlock",
        ),
        mid_block_type: str = "UNetMidBlockCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock",
            "CrossAttnUpBlock",
            "CrossAttnUpBlock",
            "CrossAttnUpBlock"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        
        use_inflated_groupnorm=False,
        use_3d=False,
        
        # Additional
        use_motion_module              = False,
        motion_module_resolutions      = ( 1,2,4,8 ),
        motion_module_mid_block        = False,
        motion_module_decoder_only     = False,
        motion_module_type             = None,
        motion_module_kwargs           = {},
        unet_use_cross_frame_attention = False,
        unet_use_temporal_attention    = False,
    ):
        super().__init__()
        
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        use_inflated_groupnorm = True if use_3d else False

        # input
        if use_3d:
            self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        else:
            self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2 ** i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_3d = use_3d,
                
                use_motion_module=use_motion_module and (res in motion_module_resolutions) and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlockCrossAttn":
            self.mid_block = UNetMidBlockCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_3d = use_3d,

                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        
        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_3d = use_3d,

                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        if use_3d:
            self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        else:
            self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,

        # support controlnet
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,

        return_dict: bool = True,
    ) -> Union[UNetConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

            down_block_res_samples += res_samples

        # support controlnet
        down_block_res_samples = list(down_block_res_samples)
        if down_block_additional_residuals is not None:
            for i, down_block_additional_residual in enumerate(down_block_additional_residuals):
                if down_block_additional_residual.dim() == 4: # boardcast
                    down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
                down_block_res_samples[i] = down_block_res_samples[i] + down_block_additional_residual

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        # support controlnet
        if mid_block_additional_residual is not None:
            if mid_block_additional_residual.dim() == 4: # boardcast
                mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
            sample = sample + mid_block_additional_residual

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size, encoder_hidden_states=encoder_hidden_states,
                )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNetConditionOutput(sample=sample)


# from mm_utils.utils import *
# model = UNetConditionModel(use_3d=False, sample_size=64, cross_attention_dim=768, **unet_additional_kwargs)
# model.load_state_dict(torch.load(os.path.join("/home/haibo/weights/stable-diffusion-v1-5", 'unet/unet.pth'), map_location='cpu'))
# print(get_parameter_number(model))

# sample = torch.randn(2, 4, 64, 64)
# encoder_hidden_states = torch.randn(2, 77, 768)
# timestep = torch.tensor([200, 300])
# y = model(sample, timestep, encoder_hidden_states, return_dict=False)[0]
# print(y.shape)