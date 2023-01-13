# -*- coding:utf-8 -*-
# email:
# create: @time: 1/7/23 10:33
import torch
from diffusers.models.attention import CrossAttention


def create_custom_diffusion(unet, freeze_model):
    for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                params.requires_grad = True
                # print(name)
            else:
                params.requires_grad = False
        elif freeze_model == 'crossattn_kv':
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                params.requires_grad = True
                # print(name)
            else:
                params.requires_grad = False
        elif freeze_model == 'none':
            params.requires_grad = False

    def new_forward(self, hidden_states, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        crossattn = False
        context = None
        if 'context' in kwargs:
            context = kwargs['context']
        elif 'encoder_hidden_states' in kwargs:
            context = kwargs['encoder_hidden_states']
        if context is not None:
            crossattn = True

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        if crossattn:
            modifier = torch.ones_like(key)
            # print(key.shape)
            modifier[:, :1, :] = modifier[:, :1, :] * 0.
            key = modifier * key + (1 - modifier) * key.detach()
            value = modifier * value + (1 - modifier) * value.detach()

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    def change_forward(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer)

    change_forward(unet)
    return unet
