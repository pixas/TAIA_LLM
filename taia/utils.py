import torch 
import torch.nn as nn 

import torch.nn.functional as F 
from copy import deepcopy
import json
# from petrel_client.client import Client
import io
import os 
from functools import wraps

def proxy_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ori_http_proxy = os.environ.get('http_proxy')  # 获取原始的http_proxy值
        ori_https_proxy = os.environ.get("https_proxy")
        os.environ['http_proxy'] = ''  # 在函数执行前将http_proxy设为空字符串
        os.environ['https_proxy'] = ''
        result = func(*args, **kwargs)  # 执行函数
        os.environ['http_proxy'] = ori_http_proxy if ori_http_proxy is not None else ''  # 函数执行后恢复原始的http_proxy值
        os.environ['https_proxy'] = ori_https_proxy if ori_https_proxy is not None else ''
        return result
    return wrapper

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

if __name__ == "__main__":
    data = client.read_jsonl("s3://syj_test/test.jsonl")
    print(data, type(data), type(data[0]))