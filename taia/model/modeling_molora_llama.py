from math import isnan
from transformers.models.llama.configuration_llama import LlamaConfig

from taia.model.modeling_molora_qwen import MoLoRAQwenDecoderLayer
from .utils import MoLoRALinear

from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
import torch.nn as nn
import torch 
import torch.nn.functional as F 
from typing import Optional, Tuple, Union, List
import warnings
from transformers.utils import logging
from dataclasses import dataclass
from copy import deepcopy

logger = logging.get_logger(__name__)
def calc_attention_fluctuation(attention_maps: torch.Tensor, consider_trace=True) -> torch.Tensor:
    # L: layer number, N: sequence length
    bs, heads, m, N = attention_maps.shape
    attention_maps = attention_maps.squeeze(0)  # (h, m, N)
    # assert attention_maps.shape == (heads, m, N)
    normalized_attention_maps = attention_maps.to(torch.float32)
    entropy = -(normalized_attention_maps * torch.log2(normalized_attention_maps + 1e-9)).sum(dim=-1)  # (h, m)
    # assert torch.isnan(entropy).sum() == 0
    # obtain weights
    if consider_trace:
        if m > 1:
            # process input
            weights = torch.arange(1, N + 1, dtype=torch.float32).to(attention_maps.device)
        else:
            # process generated tokens
            weights = N
    else:
        weights = torch.arange(0, N, dtype=torch.float32).to(attention_maps.device)
    

    if isinstance(weights, torch.Tensor):
        layer_weighted_entropy = (entropy * weights).sum(dim=-1) / weights.sum()  # (h)
    else:
        layer_weighted_entropy = entropy.mean(dim=-1)  # (h)
    
    
    return layer_weighted_entropy.mean(dim=-1)  # 1


class MoLoRALlamaConig(LlamaConfig):
    def __init__(self, vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None, hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=0.000001, use_cache=True, pad_token_id=None, bos_token_id=1, eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, rope_theta=10000, rope_scaling=None, attention_bias=False, attention_dropout=0, **kwargs):
        super().__init__(vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps, use_cache, pad_token_id, bos_token_id, eos_token_id, pretraining_tp, tie_word_embeddings, rope_theta, rope_scaling, attention_bias, attention_dropout, **kwargs)
        

@dataclass
class BaseModelOutputWithPastLogitLoss(ModelOutput):
    """
    Base class for model's outputs, with past key value states and logit bias.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    logit_loss: Optional[torch.FloatTensor] = None
    attn_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None 
    mlp_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


class MoLoRALlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        # params = {
        #     "r": config.r,
        #     "lora_alpha": config.lora_alpha,
        #     "lora_dropout": config.lora_dropout,
        #     "num_experts": config.num_experts,
        #     "num_experts_per_token": config.num_experts_per_token,
        #     "share_expert": getattr(config, "share_expert", False),
        #     "expert_sampling": True if config.expert_selection == 'sampling' else False,
        #     "use_rslora": getattr(config, "use_rslora", False),
        #     "use_logit_sum": getattr(config, "output_logit_loss", False),
        # }
        self.output_logit_loss = getattr(config, "output_logit_loss", False)
        self.ada_output_molora = getattr(config, "ada_output_molora", False)
        # 1 for absolute loss, 2 for relative loss
        self.LOGIT_GROUPING = {
            1: lambda x, y, z: (x + y + z) / 3,
            2: lambda x, y, z: torch.stack([x, y, z], dim=0),
            3: lambda x, y, z: torch.stack([x, y, z], dim=0),
        }
        # self.gate_proj = MoLoRALinear(self.hidden_size, self.intermediate_size, bias=False,
        #                               **params)
        # self.up_proj = MoLoRALinear(self.hidden_size, self.intermediate_size, bias=False, **params)
        # self.down_proj = MoLoRALinear(self.intermediate_size, self.hidden_size, bias=False, **params)
    
    def forward(self, x):
        if self.output_logit_loss:
            gate_output, gate_logit_sum = self.gate_proj(x)
            up_output, up_logit_sum = self.up_proj(x)
            down_output, down_logit_sum = self.down_proj(self.act_fn(gate_output) * up_output)
            # NOTE: current stack the logit sum
            grouping_func = self.LOGIT_GROUPING[self.output_logit_loss]
            logit_sum = grouping_func(gate_logit_sum, up_logit_sum, down_logit_sum)

            return down_output, logit_sum
        elif self.ada_output_molora and (not self.training):
            gate_output, gate_womolora_output = self.gate_proj(x)
            up_output, up_womolora_output = self.up_proj(x)
            down_output, _ = self.down_proj(self.act_fn(gate_output) * up_output)
            _, down_womolora_output = self.down_proj(self.act_fn(gate_womolora_output) * up_womolora_output)
            # NOTE: current stack the logit sum
            

            return down_output, down_womolora_output
        else:
            return super().forward(x)

class MoLoRALlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = MoLoRALlamaMLP(config)
        self.config = config 
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_logit_loss: Optional[bool] = False,
        womolora_output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        if womolora_output is not None:
            residual = (hidden_states, womolora_output)
        else:
            residual = hidden_states
            
        womolora_hidden_states = None
        hidden_states = self.input_layernorm(hidden_states)
        if womolora_output is not None:
            womolora_past_key_value = deepcopy(past_key_value)
            womolora_output = self.input_layernorm(womolora_output)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        if womolora_output is not None:
            womolora_hidden_states, womolora_self_attn_weights, womolora_present_key_value = self.self_attn(
                hidden_states=womolora_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=womolora_past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        if womolora_output is not None:
            self_attn_fluc = calc_attention_fluctuation(self_attn_weights)
            womolora_self_attn_fluc = calc_attention_fluctuation(womolora_self_attn_weights)

            if self_attn_fluc > womolora_self_attn_fluc:
                actual_use_molora = True
                residual = residual[0]
            else:

                actual_use_molora = False
                residual = residual[1]
                # past_key_value = womolora_past_key_value
                present_key_value = womolora_present_key_value
                hidden_states = womolora_hidden_states
                self_attn_weights = womolora_self_attn_weights
        
        attn_hidde_states = hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if output_logit_loss:
            hidden_states, logit_sum = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, womolora_hidden_states = hidden_states
                womolora_hidden_states = residual + womolora_hidden_states
        mlp_hidden_states = hidden_states
        
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        if output_logit_loss:
            outputs += (logit_sum,)
        if womolora_hidden_states is not None:
            outputs += (womolora_hidden_states,)
        outputs += (attn_hidde_states, mlp_hidden_states)

        return outputs

class MoLoRALlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MoLoRALlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.ada_output_molora = getattr(config, "ada_output_molora", False)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_logit_loss: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_logit_loss = output_logit_loss if output_logit_loss is not None else getattr(self.config, "output_logit_loss", 0)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds
        
        womolora_hidden_states = None 

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_attn_hidden_states = () if output_hidden_states else None 
        all_mlp_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_logit_loss else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    output_logit_loss,
                    womolora_hidden_states
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    output_logit_loss=output_logit_loss,
                    womolora_output=womolora_hidden_states
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_logit_loss:
                all_router_logits += (layer_outputs[-2],)
            if self.ada_output_molora:
                womolora_hidden_states = layer_outputs[-1]
            if output_hidden_states:
                all_attn_hidden_states += (layer_outputs[-2],)
                all_mlp_hidden_states += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits] if v is not None)
        return BaseModelOutputWithPastLogitLoss(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            logit_loss=all_router_logits,
            attn_hidden_states=all_attn_hidden_states,
            mlp_hidden_states=all_mlp_hidden_states
        )

class MoLoRALlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MoLoRALlamaModel(config)