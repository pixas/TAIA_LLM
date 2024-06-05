from numpy import add
import torch 
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from peft.utils import _get_submodules
# from peft.tuners.lora import mark_only_lora_as_trainable
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Union, Tuple


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)
        
def mark_only_lora_as_trainable(model: nn.Module, bias, skip_experts) -> None:
    num_experts, add_identity_mapping = skip_experts
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad = False
        if "lora" in n and any([f"experts.{num_experts + i}" in n for i in range(add_identity_mapping)]):
            p.requires_grad = False

    if bias == "none":
        return

    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

def check_target_module_exists(lora_config, key, target_modules):
    target_module_found = any(key.endswith(module_name) for module_name in target_modules)
    return target_module_found

def create_mixoflora_module(lora_config, target, num_experts, num_experts_per_token, expert_sampling, use_logit_sum=False, add_bias=True, add_identity_mapping = 0, ada_output_molora=False):
    in_features, out_features = target.in_features, target.out_features
    new_module = MoLoRALinear(in_features, out_features, num_experts, num_experts_per_token,
                              r=lora_config.molora_r,
                              lora_alpha=lora_config.molora_alpha,
                              lora_dropout=lora_config.lora_dropout,
                              use_rslora=lora_config.use_rslora,
                              expert_sampling=expert_sampling,
                              use_logit_sum=use_logit_sum,
                              add_identity_mapping=add_identity_mapping,
                              ada_output_molora=ada_output_molora,
                              bias=add_bias)
    return new_module


def get_mixoflora_model(model, lora_config,  **kwargs):
    num_experts = kwargs.pop("num_experts", 1)
    num_experts_per_token = kwargs.pop("num_experts_per_token", 1)
    expert_selection = kwargs.pop("expert_selection", "topk")
    use_logit_sum = kwargs.pop("use_logit_sum", 0)
    decoder_type = kwargs.pop("decoder_type", Qwen2DecoderLayer)
    inference_mode = kwargs.pop("inference_mode", False)
    add_identity_mapping = kwargs.pop("add_identity_mapping", 0)
    ada_output_molora = kwargs.pop("ada_output_molora", False)
    wrap_modules = kwargs.pop("wrap_modules", ("mlp"))

    # find linear modules with "switch" in their attributes
    key_list = [key for key, _ in model.named_modules()]
    target_module_names = set()

    for name, module in model.named_modules():
        rank0_print(name, module)
        if isinstance(module, torch.nn.Linear):
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), decoder_type) and any(module in name for module in wrap_modules):
                names = name.split(".")
                target_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_module_names = list(target_module_names)
    for key in key_list:
        if not check_target_module_exists(lora_config, key, target_module_names):
            continue
            
        parent, target, target_name = _get_submodules(model, key)
        # print(parent, target_name)
        if hasattr(target, "bias"):
            if target.bias is not None:
                add_bias = True 
            else: 
                add_bias = False
        else:
            add_bias = False
        new_module = create_mixoflora_module(
            lora_config, 
            target, 
            num_experts, 
            num_experts_per_token, 
            True if expert_selection == "sampling" else False, 
            use_logit_sum=use_logit_sum, 
            add_bias=add_bias,
            add_identity_mapping=add_identity_mapping,
            ada_output_molora=ada_output_molora)
        setattr(parent, target_name, new_module)
        new_module.weight = target.weight 
        if hasattr(target, "bias"):
            if target.bias is not None:
                new_module.bias = target.bias

        new_module.to(target.weight.device)
        
        if getattr(target, "state", None) is not None:
            new_module.state = target.state
            new_module.to(target.weight.device)
        
        del target
    if not inference_mode:
        mark_only_lora_as_trainable(model, getattr(lora_config, "bias", "none"), skip_experts=(num_experts, add_identity_mapping))
    if inference_mode:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_identity_mapping()
    else:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_parameters()
    
    return model


def get_part_mixoflora_model(model, lora_config,  **kwargs):
    num_experts = kwargs.pop("num_experts", 1)
    num_experts_per_token = kwargs.pop("num_experts_per_token", 1)
    expert_selection = kwargs.pop("expert_selection", "topk")
    use_logit_sum = kwargs.pop("use_logit_sum", 0)
    decoder_type = kwargs.pop("decoder_type", Qwen2DecoderLayer)
    inference_mode = kwargs.pop("inference_mode", False)
    add_identity_mapping = kwargs.pop("add_identity_mapping", 0)
    add_layer_index = kwargs.pop("add_layer_index", 0)
    ada_output_molora = kwargs.pop("ada_output_molora", False)
    wrap_modules = kwargs.pop("wrap_modules", ("mlp"))
    # find linear modules with "switch" in their attributes
    key_list = [key for key, _ in model.named_modules()]
    target_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), decoder_type) and any(module in name for module in wrap_modules):
                names = name.split(".")
                target_module_names.add(names[0] if len(names) == 1 else names[-1])
    target_module_names = list(target_module_names)
    
    for key in key_list:
        if not check_target_module_exists(lora_config, key, target_module_names):
            continue
        
        # get layer from models.layers.xx.mlp
        layer = int(key.split(".")[2])
        if layer < add_layer_index:
            continue
        parent, target, target_name = _get_submodules(model, key)

        if hasattr(target, "bias"):
            if target.bias is not None:
                add_bias = True 
            else: 
                add_bias = False
        else:
            add_bias = False
        new_module = create_mixoflora_module(
            lora_config, 
            target, 
            num_experts, 
            num_experts_per_token, 
            True if expert_selection == "sampling" else False, 
            use_logit_sum=use_logit_sum, 
            add_bias=add_bias,
            add_identity_mapping=add_identity_mapping,
            ada_output_molora=ada_output_molora)
        setattr(parent, target_name, new_module)
        new_module.weight = target.weight 
        if hasattr(target, "bias"):
            if target.bias is not None:
                new_module.bias = target.bias

        new_module.to(target.weight.device)
        
        if getattr(target, "state", None) is not None:
            new_module.state = target.state
            new_module.to(target.weight.device)
        
        del target
    if not inference_mode:
        mark_only_lora_as_trainable(model, getattr(lora_config, "bias", "none"), skip_experts=(num_experts, add_identity_mapping))
    if inference_mode:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_identity_mapping()
        # for n, p in model.named_parameters():
        #     if "lora" in n:
        #         p.requires_grad = False
    else:
        for n, m in model.named_modules():
            if isinstance(m, MoLoRALinear):
                m.reset_parameters()
    
    return model

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRAModule, self).__init__()
        self.lora_a = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_b = nn.Parameter(torch.zeros((out_features, r)))
        self.reset_parameters()

    def forward(self):
        return self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

class Identity(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = in_features 
        self.out_features = out_features
    
    def forward(self, x):
        return torch.zeros(x.shape[:-1] + (self.out_features, )).to(x)

class MoLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        use_logit_sum: int = 0,
        use_lbl_loss: bool = False,
        share_expert: bool = False,
        expert_sampling: bool = False,
        use_rslora: bool = False,
        ada_output_molora: bool = False,
        add_identity_mapping: int = 0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # moe parameters
        self.num_experts = num_experts 
        self.num_experts_per_token = num_experts_per_token
        self.share_expert = share_expert
        self.expert_sampling = expert_sampling
        self.use_rslora = use_rslora
        self.add_identity_mapping = add_identity_mapping
        self.ada_output_molora = ada_output_molora
        
        if self.share_expert:
            self.num_experts_per_token -= 1
        self.use_logit_sum = use_logit_sum
        if num_experts > 1:
            if self.use_logit_sum:
                if False:
                    num_choices = math.comb(self.num_experts, self.num_experts_per_token)
                    self.switch = nn.Linear(in_features, num_choices)
                else:
                    self.switch = nn.Linear(in_features, num_experts + add_identity_mapping)
            else:
                self.switch = nn.Linear(in_features, num_experts + add_identity_mapping)
        self.use_lbl_loss = use_lbl_loss    
        
        # Actual trainable parameters
        if r > 0:
            # self.experts = nn.ModuleList([
            #     nn.ModuleDict({"lora_A_{}".format(i): nn.Linear(in_features, r, False, dtype=torch.float32),
            #                    "lora_B_{}".format(i): nn.Linear(r, out_features, False, dtype=torch.float32)}) if i < self.num_experts else Identity(in_features, out_features)
            # for i in range(self.num_experts + self.add_identity_mapping) ])
            self.experts = nn.ModuleList([
                nn.ModuleDict({"lora_A_{}".format(i): nn.Linear(in_features, r, False, dtype=torch.float32),
                               "lora_B_{}".format(i): nn.Linear(r, out_features, False, dtype=torch.float32)}) 
            for i in range(self.num_experts + self.add_identity_mapping) ])


            self.scaling = self.lora_alpha / (math.sqrt(self.r) if self.use_rslora else self.r)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_identity_mapping(self):
        if hasattr(self, 'experts'):
           
            for idx, expert in enumerate(self.experts):
                if idx < self.num_experts:
                    pass
                else:
                    nn.init.zeros_(expert[f'lora_A_{idx}'].weight)
                    nn.init.zeros_(expert[f'lora_B_{idx}'].weight)

    
    def reset_parameters(self):
        # print(self.weight.shape)
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, 'experts'):
        
            for idx, expert in enumerate(self.experts):
                if idx < self.num_experts:
                    nn.init.kaiming_uniform_(expert[f'lora_A_{idx}'].weight, a=math.sqrt(5))
                    nn.init.zeros_(expert[f'lora_B_{idx}'].weight)
                else:
                    nn.init.zeros_(expert[f'lora_A_{idx}'].weight)
                    nn.init.zeros_(expert[f'lora_B_{idx}'].weight)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)


    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.use_lbl_loss:
                moe_result, lbl_loss = self.molora_helper2(x)
                return result + moe_result, lbl_loss
            elif self.use_logit_sum:
                moe_result, logit_sum = self.molora_helper2(x) if self.training else self.molora_helper(x)
                result += moe_result
                return result, logit_sum
            else:
                moe_result = self.molora_helper2(x) if self.training else self.molora_helper(x)
                if self.ada_output_molora:
                    return result + moe_result, result 
                else:
                    return result + moe_result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    
    def molora_helper2(self, x: torch.Tensor):
        if self.num_experts <= 1:
            previous_dtype = x.dtype 
            x = x.to(self.experts[0]['lora_A_0'].weight.dtype)
            expert_output = self.experts[0]['lora_B_0'](
                self.experts[0]['lora_A_0'](self.lora_dropout(x))
            ) * self.scaling
            expert_output = expert_output.to(previous_dtype)
            return expert_output
        
        previous_dtype = x.dtype 
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)
        x = x.to(self.experts[0]['lora_A_0'].weight.dtype)
        if self.share_expert:
            share_result = self.experts[0][f'lora_B_0'](self.experts[0][f'lora_A_0'](x)) * self.scaling
        gate_logits = self.switch(x)  # [bs * N, expert]
        # x = self.lora_dropout(x)
        if self.share_expert:
            temp_results = torch.stack([expert[f'lora_B_{i}'](expert[f'lora_A_{i}'](x)) * self.scaling for i, expert in enumerate(self.experts[1:])], dim=0)  # [expert, bs * N, out_features]
        else:
            temp_results = torch.stack([expert[f'lora_B_{i}'](expert[f'lora_A_{i}'](x)) * self.scaling for i, expert in enumerate(self.experts)], dim=0)  # [expert, bs * N, out_features]
        temp_results = temp_results.transpose(0, 1)  # [bs * N, expert, out_features]
        if self.expert_sampling:
            # 根据gate logits的概率分布，选择expert
            gate_logit_prob = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
            # find the element that is not in the range [0, 1]

            assert torch.all(gate_logit_prob >= 0) and torch.all(gate_logit_prob <= 1)
            selected_experts = torch.multinomial(gate_logit_prob, self.num_experts_per_token, replacement=False)
            # 把gate_logits[:, self.num_experts:]的部分设为-infinity
            mask_gate_logits = gate_logits.clone().to(torch.float32)
            mask_gate_logits[:, self.num_experts:] = -1e4
            weights = torch.gather(mask_gate_logits, 1, selected_experts)
            if self.use_logit_sum:
                gate_logit_sum = torch.log(gate_logit_prob)
                logit_sum = torch.gather(gate_logit_sum, 1, selected_experts).sum(dim=-1)
                
        else:
            mask_gate_logits = gate_logits.clone()
            mask_gate_logits[:, self.num_experts:] = -1e4
            weights, selected_experts = torch.topk(mask_gate_logits, self.num_experts_per_token)
        # weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)
        # given a tensor with shape [b, N, d] and a index tensor [b, k]
        # how to obtain a tensor with shape [b, k, d]?
        
        selected_results = temp_results.gather(1, selected_experts.unsqueeze(-1).expand(-1, -1, self.out_features))  # [bs * N, select_expert, out_features]
        assert selected_results.shape == (batch_size * N, self.num_experts_per_token, self.out_features)
        if self.share_expert:
            weights = torch.cat([weights, torch.ones(weights.shape[0], 1).to(weights)], dim=-1)
            selected_results = torch.cat([
                selected_results,
                share_result.unsqueeze(1)
            ], dim=1)

        
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(selected_results.dtype)  # [bs * N, expert]
        results = torch.einsum("be, bef -> bf", weights, selected_results)
        results = results.contiguous().view(batch_size, N, -1)
        results = results.to(previous_dtype)
        if self.use_logit_sum:
            logit_sum = logit_sum.to(previous_dtype)
            # multiply the last dimension of weights
            # logit_sum = weights[..., 0] * weights[..., 1] \cdots weights[..., -1]
            # logit_sum = torch.prod(weights, dim=-1)
            # logit_sum = torch.sum(torch.log(weights, dim=-1), dim=-1)
            return results, logit_sum
        else:
            return results
        
    
    def molora_helper(self, x: torch.Tensor):
        # debug:
        if self.num_experts <= 1:
            expert_output = self.experts[0]['lora_B_0'](
                self.experts[0]['lora_A_0'](self.lora_dropout(x))
            ) * self.scaling
            return expert_output
        batch_size, N, d = x.shape 
        # if N == 1:
        #     return 0
        previous_dtype = x.dtype
        x = x.contiguous().view(-1, d)       
        gate_logits = self.switch(x)  # [bs * N, expert]
        
        # selected experts: 选中的最大的两个概率的expert的编号
        if self.expert_sampling and self.training:
            # 根据gate logits的概率分布，选择expert
            gate_logit_prob = F.softmax(gate_logits, dim=-1, dtype=torch.float32)
            # find the element that is not in the range [0, 1]

            assert torch.all(gate_logit_prob >= 0) and torch.all(gate_logit_prob <= 1)
            selected_experts = torch.multinomial(gate_logit_prob, self.num_experts_per_token, replacement=False)
            # 把gate_logits[:, self.num_experts:]的部分设为-infinity
            mask_gate_logits = gate_logits.clone().to(torch.float32)
            mask_gate_logits[:, self.num_experts:] = -1e4
            weights = torch.gather(mask_gate_logits, 1, selected_experts)

        else:
            mask_gate_logits = gate_logits.clone()
            mask_gate_logits[:, self.num_experts:] = -1e4
            weights, selected_experts = torch.topk(mask_gate_logits, self.num_experts_per_token)
        weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        x = x.to(self.experts[0]['lora_A_0'].weight.dtype)
        results = torch.zeros((batch_size * N, self.out_features)).to(x) # bs*N, d
        load_balancing_loss = 0

        if self.training or N > 1:
            for i, expert in enumerate(self.experts):
                
                # batch_idx: batch端的下标
                # nth_expert: 不是expert的下标，而是对应selected_experts[batch_idx, nth_expert]的expert的下标
                
                batch_idx, nth_expert = torch.where(selected_experts == i) 
                # batch_idx: [bs * N, 1]
                # nth_expert: [bs * N, 1]

                expert_output = expert['lora_B_{}'.format(i)](
                    expert['lora_A_{}'.format(i)](self.lora_dropout(x[batch_idx]))
                ) * self.scaling # if i < self.num_experts else expert(self.lora_dropout(x[batch_idx]))
                # expert_output = expert(x[batch_idx])
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_output
                # begin to compute load balancing loss 
                # compute the number of tokens routed to each expert
                # compute the fraction of tokens routed to each expert
                # 选择第i个expert的token数量
                num_per_expert = len(batch_idx)
                # 选择第i个expert的token 比例，对应公式中的f_i
                fraction_per_expert = num_per_expert / (batch_size * N)
                # # 选择第i个expert的所有token的概率的均值，对应公式中的P_i
                prob_per_expert = weights[batch_idx, nth_expert, None].mean()
                load_balancing_loss += fraction_per_expert * prob_per_expert
            load_balancing_loss = load_balancing_loss * self.num_experts / (self.num_experts_per_token * self.num_experts_per_token)
        else:
            assert selected_experts.shape[0] == 1
            
            selected_experts = selected_experts.flatten()
            weights = weights.flatten()
            for idx, expert_idx in enumerate(selected_experts):
                results += weights[idx] * (self.experts[expert_idx]['lora_B_{}'.format(expert_idx)](
                    self.experts[expert_idx]['lora_A_{}'.format(expert_idx)](self.lora_dropout(x))
                ) * self.scaling )
                    #if expert_idx < self.num_experts else self.experts[expert_idx](self.lora_dropout(x)))
        
        results = results.contiguous().view(batch_size, N, self.out_features)
        results = results.to(previous_dtype)
        if self.use_lbl_loss:
            return results, load_balancing_loss
        elif self.use_logit_sum:
            return_logit_sum = torch.zeros((batch_size * N, 1)).to(x)
            return results, return_logit_sum
        else:
            return results