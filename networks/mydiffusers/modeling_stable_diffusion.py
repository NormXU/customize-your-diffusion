# -*- coding:utf-8 -*-
# email:
# create: @time: 12/14/22 10:24
import os
import json
import logging
import torch.nn as nn
from diffusers.configuration_utils import FrozenDict
from diffusers import AutoencoderKL, UNet2DConditionModel
from .ema import EMAModel

class StableDiffusion(nn.Module):
    def __init__(self, pretrained_model_name_or_path,
                 tokenizer,
                 text_encoder,
                 revision=None,
                 gradient_checkpointing=False, apply_xformers=False,
                 use_ema=False,
                 **kwargs):
        super().__init__()

        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
        )
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        if gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        if apply_xformers:
            try:
                unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logging.warning(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        self.vae = vae
        self.unet = unet
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.ema_unet = None

        if use_ema:
            self.ema_unet = EMAModel(unet.parameters())

        schedule_config_pth = os.path.join(pretrained_model_name_or_path,
                                           "scheduler/scheduler_config.json")
        with open(schedule_config_pth, "r", encoding="utf-8") as reader:
            text = reader.read()
        self.scheduler_config = FrozenDict(json.loads(text))
