# -*- coding:utf-8 -*-
# email:
# create: 2021/7/15
# region for master
import os
from .mydiffusers.modeling_stable_diffusion import StableDiffusion
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor


# endregion


def get_network(model_args):
    model_type = model_args.pop("type")
    if model_type == "StableDiffusion":
        tokenizer_args = model_args.pop("tokenizer_args")
        text_model_args = model_args.pop("text_model_args")
        pretrained_model_name_or_path = model_args["pretrained_model_name_or_path"]
        tokenizer = prepare_clip_tokenizer(tokenizer_args, pretrained_model_name_or_path)
        text_encoder = prepare_clip_text_encoder(text_model_args, pretrained_model_name_or_path)
        return eval(model_type)(tokenizer=tokenizer,
                                text_encoder=text_encoder,
                                **model_args)
    else:
        return eval(model_type)(**model_args)

def prepare_clip_tokenizer(tokenizer_args, pretrained_model_name_or_path=None):
    tokenizer_type = tokenizer_args["type"]
    tokenizer_pretrained_model_path = tokenizer_args.get("pretrained_model_name_or_path", None)
    if tokenizer_pretrained_model_path is None:
        tokenizer_pretrained_model_path = os.path.join(pretrained_model_name_or_path, "tokenizer")
    tokenizer = eval(tokenizer_type).from_pretrained(
        tokenizer_pretrained_model_path,
    )
    if isinstance(tokenizer, CLIPProcessor):
        tokenizer = tokenizer.tokenizer
    return tokenizer


def prepare_clip_text_encoder(text_model_args, pretrained_model_name_or_path=None):
    # prepare CLIP text encoder
    text_model_type = text_model_args["type"]
    text_model_pretrained_path = text_model_args.get("pretrained_model_name_or_path", None)
    if text_model_pretrained_path is None:
        text_model_pretrained_path = os.path.join(pretrained_model_name_or_path, "text_encoder")
    text_encoder = eval(text_model_type).from_pretrained(
        text_model_pretrained_path,
    )
    return text_encoder
