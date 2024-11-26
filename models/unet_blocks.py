# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from models.resnet import Downsample, ResnetBlock, Upsample
from models.attention import TransformerModel
from models.motion_module import get_motion_module

def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    
    unet_use_cross_frame_attention=False,
    unet_use_temporal_attention=False,
    use_inflated_groupnorm=False,
    use_3d = False,
    use_motion_module=None,
    
    motion_module_type=None,
    motion_module_kwargs=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock":
        return DownBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,

            use_3d = use_3d,
            use_inflated_groupnorm=use_inflated_groupnorm,

            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )
    elif down_block_type == "CrossAttnDownBlock":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock")
        return CrossAttnDownBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,

            use_3d = use_3d,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            
            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    mid_block_only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = 1,
    dropout: float = 0.0,
    use_3d: bool = False,
):
    if mid_block_type == "UNetMidBlockCrossAttn":
        return UNetMidBlockCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
            use_3d=use_3d,
        )
    elif mid_block_type is None:
        return None
    else:
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",

    unet_use_cross_frame_attention=False,
    unet_use_temporal_attention=False,
    use_inflated_groupnorm=False,
    
    use_3d = False,

    use_motion_module=None,
    motion_module_type=None,
    motion_module_kwargs=None,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock":
        return UpBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,

            use_inflated_groupnorm=use_inflated_groupnorm,
            use_3d = use_3d,

            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )
    elif up_block_type == "CrossAttnUpBlock":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,

            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,

            use_3d = use_3d,

            use_motion_module=use_motion_module,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,

        use_inflated_groupnorm=False,
        
        use_3d=False,

        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()

        resnets = []
        motion_modules = []
        use_inflated_groupnorm = True if use_3d else False

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,

                    use_3d = use_3d,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            if use_3d and use_motion_module:
                motion_modules.append(
                    get_motion_module(
                        in_channels=out_channels,
                        motion_module_type=motion_module_type, 
                        motion_module_kwargs=motion_module_kwargs,
                    )
                )
            
        self.resnets = nn.ModuleList(resnets)
        if use_motion_module and use_3d:
            self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op", use_3d=use_3d
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()
        if hasattr(self, "motion_modules"):
            for resnet, motion_module in zip(self.resnets, self.motion_modules):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                output_states += (hidden_states,)
        else:
            for resnet in self.resnets:
                hidden_states = resnet(hidden_states, temb)
                output_states += (hidden_states,)


        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,

        use_3d = False,
        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False,
        use_inflated_groupnorm=False,
        
        use_motion_module=None,

        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        self.use_3d = use_3d
        self.use_motion_module = use_motion_module
        resnets = []
        attentions = []
        motion_modules = []
        use_inflated_groupnorm = True if use_3d else False

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_3d=use_3d,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

            attentions.append(
                TransformerModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_3d=use_3d,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )

            if use_3d and use_motion_module:
                motion_modules.append(
                    get_motion_module(
                        in_channels=out_channels,
                        motion_module_type=motion_module_type, 
                        motion_module_kwargs=motion_module_kwargs,
                    )
                )
            
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if use_motion_module and use_3d:
            self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op", use_3d=use_3d,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None):
        output_states = ()
        if hasattr(self, "motion_modules"):
            for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                output_states += (hidden_states,)
        else:
            for resnet, attn in zip(self.resnets, self.attentions):
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class UNetMidBlockCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,

        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False,
        use_inflated_groupnorm=False,

        use_motion_module=None,
        use_3d = False,
        
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        if use_3d:
            use_inflated_groupnorm = True

        # there is always at least one resnet
        resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,

                use_3d = use_3d,
                use_inflated_groupnorm=use_inflated_groupnorm,
            )
        ]
        attentions = []
        motion_modules = []

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                TransformerModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,

                    use_3d = use_3d,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            if use_3d and use_motion_module:
                motion_modules.append(
                    get_motion_module(
                        in_channels=in_channels,
                        motion_module_type=motion_module_type, 
                        motion_module_kwargs=motion_module_kwargs,
                    )
                )
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,

                    use_3d = use_3d,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if use_motion_module and use_3d:
            self.motion_modules = nn.ModuleList(motion_modules)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        if hasattr(self, "motion_modules"):
            for attn, resnet, motion_module in zip(self.attentions, self.resnets[1:], self.motion_modules):
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
                hidden_states = resnet(hidden_states, temb)
        else:
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                hidden_states = resnet(hidden_states, temb) 

        return hidden_states


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,

        use_inflated_groupnorm=False,
        use_3d = False,

        use_motion_module=None,
        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        if use_3d:
            use_inflated_groupnorm = True

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    use_3d=use_3d,
                    use_inflated_groupnorm=use_inflated_groupnorm,
                )
            )
            if use_3d and use_motion_module:
                motion_modules.append(
                    get_motion_module(
                        in_channels=out_channels,
                        motion_module_type=motion_module_type, 
                        motion_module_kwargs=motion_module_kwargs,
                    )
                )

        self.resnets = nn.ModuleList(resnets)
        if use_3d and use_motion_module:
            self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels, use_conv=True, out_channels=out_channels, use_3d=use_3d,)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, encoder_hidden_states=None,):
        
        if hasattr(self, "motion_modules"):
            for resnet, motion_module in zip(self.resnets, self.motion_modules):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
        else:
            for resnet in self.resnets:
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class CrossAttnUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,

        unet_use_cross_frame_attention=False,
        unet_use_temporal_attention=False,
        use_inflated_groupnorm=False,
        
        use_motion_module=None,
        use_3d = False,

        motion_module_type=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        if use_3d:
            use_inflated_groupnorm = True
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,

                    use_inflated_groupnorm=use_inflated_groupnorm,
                    use_3d=use_3d,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                TransformerModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,

                    use_3d=use_3d,

                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
            )
            if use_3d and use_motion_module:
                motion_modules.append(
                    get_motion_module(
                        in_channels=out_channels,
                        motion_module_type=motion_module_type, 
                        motion_module_kwargs=motion_module_kwargs,
                    )
                )
            
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if use_3d and use_motion_module:
            self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample(out_channels, use_conv=True, out_channels=out_channels, use_3d=use_3d)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
    ):
        if hasattr(self, "motion_modules"):
            for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                # add motion module
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states)
        else:
            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


# in_channels = 1280
# sample_size = 8
# batch_size = 2
# frame_num = 16

# temb_channels = 1280
# out_channels = in_channels
# num_layers = 1
# add_downsample = True

# cross_attention_dim = 768

# use_3d = False
# unet_additional_kwargs = {
#     'use_inflated_groupnorm': True, 
#     'use_motion_module': True, 
#     'motion_module_type': 'Vanilla', 
#     'motion_module_kwargs': {
#         'num_attention_heads': 8, 
#         'num_transformer_block': 1, 
#         'attention_block_types': ['Temporal_Self', 'Temporal_Self'], 
#         'temporal_position_encoding': True, 
#         'temporal_attention_dim_div': 1, 
#         'zero_initialize': True
#         }
#     }


# if use_3d:
#     x = torch.randn(batch_size, in_channels, frame_num, sample_size, sample_size)
# else:
#     x = torch.randn(batch_size, in_channels, sample_size, sample_size)
# t = torch.randn(batch_size, temb_channels)
# z = torch.randn(batch_size, 77, cross_attention_dim)
# model = UNetMidBlockCrossAttn(use_3d=use_3d, num_layers=num_layers, in_channels=in_channels, temb_channels=temb_channels, cross_attention_dim=cross_attention_dim, attn_num_head_channels=8, **unet_additional_kwargs)

# y = model(x, t, z)
# print(model)
# print(y.shape)
