#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
from copy import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from taia.model.utils import get_mixoflora_model, get_part_mixoflora_model
from taia.model import MoLoRAQwenForCausalLM, MoLoRALlamaForCausalLM, MoLoRAQwenDecoderLayer, MoLoRALlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import json
from safetensors.torch import load_file
import re 


def get_model_class_from_path(model_path: str):
    model_path = model_path.lower()
    if "qwen" in model_path or "ming" in model_path:
        return MoLoRAQwenForCausalLM, Qwen2DecoderLayer
    if "llama" in model_path:
        return MoLoRALlamaForCausalLM, LlamaDecoderLayer

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, use_logit_bias=False, device_map="auto", device="cuda", unload_ffn: bool = False, return_base=False):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if model_base is not None:
        # PEFT model
        from peft import PeftModel, LoraConfig
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, trust_remote_code=True,**kwargs)
        base_model = model
        print(f"Loading LoRA weights from {model_path}")
        lora_config = LoraConfig.from_pretrained(model_path)
        if unload_ffn:
            lora_config.target_modules = ['v_proj', 'k_proj', 'q_proj', 'o_proj']
        print(lora_config)
        model = PeftModel.from_pretrained(model, model_path, config=lora_config)
        print(f"Merging weights")
        model = model.merge_and_unload()
        print('Convert to FP16...')
        model.to(torch.float16)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        if unload_ffn:
            if "1.8b" in model_path:
                pretrained_path = "/mnt/petrelfs/jiangshuyang.p/models/models--Qwen--Qwen1.5-1.8B-Chat"
            elif "7b" in model_path:
                if "ming" in model_path or "qwen" in model_path:
                    pretrained_path = "/mnt/petrelfs/jiangshuyang.p/models/models--Qwen--Qwen1.5-7B-Chat"
                else:
                    pass
            elif "8b" in model_path:
                pass
            
            print("Load pretrained weights over...")
            model = AutoModelForCausalLM.from_pretrained(pretrained_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            else:
                all_checkpoints = os.listdir(model_path)
                all_checkpoints = [x for x in all_checkpoints if x.endswith(".safetensors")]
                state_dict = {}
                for checkpoint in all_checkpoints:
                    state_dict.update(load_file(os.path.join(model_path, checkpoint)))
            filter_keys = ['gate_proj', 'down_proj', 'up_proj']
            # filter those keys whose names contains any key in filter_keys
            new_state_dict = {k: v for k, v in state_dict.items() if not any(x in k for x in filter_keys)}
            model.load_state_dict(new_state_dict, strict=False)
            # new_state_dict.update(pretrained_weight)
            print("merge pretrained weight with finetuned weight..")
            # for k in new_state_dict.keys():
            #     print(k)
            # model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, state_dict=new_state_dict, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
        model.to(torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
        if tokenizer_with_prefix_space.pad_token_id is None:
            tokenizer_with_prefix_space.pad_token_id = tokenizer_with_prefix_space.eos_token_id
    else:
        tokenizer_with_prefix_space = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
        
    if return_base:
        return tokenizer, model, context_len, tokenizer_with_prefix_space, base_model 
    return tokenizer, model, context_len, tokenizer_with_prefix_space, None


def load_molora_pretrained_model(model_path, model_base, model_name, load_molora=True, load_8bit=False, load_4bit=False, use_logit_bias=False, add_layer_index=0, ada_output_molora=False, unload_lora=True, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

        # Load language model
    if model_base is not None:
        # PEFT model
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        
        if not hasattr(lora_cfg_pretrained, "num_experts"):
            lora_cfg_pretrained.num_experts = 4
            lora_cfg_pretrained.num_experts_per_token = 2
            lora_cfg_pretrained.share_expert = False
            lora_cfg_pretrained.expert_selection = "top_k"
        if getattr(lora_cfg_pretrained, "use_rslora", None) is None:
            setattr(lora_cfg_pretrained, "use_rslora", False)
        
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            with open(os.path.join(model_path, "adapter_config.json")) as f:
                lora_specific_pretrained = json.load(f)
        else:
            lora_specific_pretrained = {
                "r": 32,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "bias": "none"
            }

        # merge lora_specific_pretrained to lora_cfg_pretrained, which is a transformer config class
        lora_cfg_pretrained.r = lora_specific_pretrained['r']
        lora_cfg_pretrained.lora_alpha = lora_specific_pretrained["lora_alpha"]
        lora_cfg_pretrained.lora_dropout = lora_specific_pretrained["lora_dropout"]
        lora_cfg_pretrained.bias = lora_specific_pretrained['bias']
        lora_cfg_pretrained.molora_r = getattr(lora_cfg_pretrained, "molora_r", lora_cfg_pretrained.r)
        lora_cfg_pretrained.molora_alpha = getattr(lora_cfg_pretrained, "molora_alpha", lora_cfg_pretrained.lora_alpha)
        
        add_identity_mapping = getattr(lora_cfg_pretrained, "add_identity_mapping", 0)
        lora_cfg_pretrained.ada_output_molora = ada_output_molora
        lora_cfg_pretrained.output_attentions = True
        wrap_modules = getattr(lora_cfg_pretrained, "wrap_modules", ['mlp'])
        
        print(lora_cfg_pretrained)
        model_cls, decoder_layer_cls = get_model_class_from_path(model_path)
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        if not unload_lora:
            from peft import PeftModel
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
        

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        
        # all_mlp_weights = [y for x, y in model.state_dict().items() if "gate_proj.weight" in x or "down_proj.weight" in x or "up_proj.weight" in x]
        print(f'Loading MoLoRA weights from {model_path}')
        if load_molora:
            model = get_part_mixoflora_model(model, lora_config=lora_cfg_pretrained,
                                        num_experts=lora_cfg_pretrained.num_experts,
                                            num_experts_per_token=lora_cfg_pretrained.num_experts_per_token,
                                            expert_selection=lora_cfg_pretrained.expert_selection,
                                            decoder_type=decoder_layer_cls,
                                            use_logit_sum=False,
                                            add_identity_mapping=add_identity_mapping,
                                            inference_mode=True,
                                            add_layer_index=add_layer_index,
                                            ada_output_molora=ada_output_molora,
                                            wrap_modules=wrap_modules)
            model.config.ada_output_molora = ada_output_molora
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            else:
                raise ValueError("A Mix-of-Lora models require non-lora-trainables.bin")
            incompatible_keys = model.load_state_dict(non_lora_trainables, strict=False)


        print('Convert to FP16...')
        print(model)
        model.to(torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    if use_logit_bias:
        if model_base is not None:
            # lora case
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base , add_prefix_space=True, trust_remote_code=True)
        else:
            tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, trust_remote_code=True)
        if tokenizer_with_prefix_space.pad_token_id is None:
            tokenizer_with_prefix_space.pad_token_id = tokenizer_with_prefix_space.eos_token_id
    else:
        tokenizer_with_prefix_space = None
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id   
    return tokenizer, model, context_len, tokenizer_with_prefix_space
