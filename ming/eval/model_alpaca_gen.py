import argparse
import torch
import os

from tqdm import tqdm
# import shortuuid
from ming.utils import client

from ming.conversations import conv_templates, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model
from ming.utils import get_model_name_from_path
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import DataLoader

import datasets 
from copy import deepcopy

import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset:
    def __init__(self, questions):
        self.questions = questions
        # self.tokenizer = tokenizer
        # self.model_config = model_config
        self.index = 0

    def __getitem__(self, index):
        line = self.questions[index]
        
        # return question, ansewr, additional info
        question = line['conversations'][0]['value']
        answer = line['conversations'][1]['value'] if len(line['conversations']) > 1 else None

        additional_info = line['eval']

        
        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return question, answer, additional_info

    def __len__(self):
        return len(self.questions)

    def __iter__(self):
        # 返回迭代器对象本身
        return self
    
    def __next__(self):
        if self.index < len(self.questions):
            # 返回下一个值并更新索引
            item = self.questions[self.index]
            self.index += 1
            return item
        else:
            # 没有更多元素时抛出StopIteration异常
            raise StopIteration


# DataLoader
def create_data_loader(questions, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def convert_to_json(questions):
    # questions is a pandas dataframe, which is to be converted to a list object
    # each element in the list is a dictionary
    # the column name of questions is the key of the dictionary
    # the value of the dictionary is the value of the corresponding column
    questions = questions.to_dict(orient='records')
    return questions

def eval_model(args):
    # Model
    # disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    test_base = model_path is None and args.model_base is not None 
    # else:
    if args.load_molora:
    # if "molora" in model_path:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_molora_pretrained_model(model_path, args.model_base, model_name, args.load_molora, use_logit_bias=args.use_logit_bias, add_layer_index=args.add_layer_index, ada_output_molora=args.ada_output_molora, unload_lora=args.unload_lora)
    else:
        tokenizer, model, context_len, tokenizer_with_prefix_space = load_pretrained_model(model_path, args.model_base, model_name, use_logit_bias=args.use_logit_bias, unload_ffn=args.unload_ffn)

    # load args.question_file, which is a csv file
    
    

    answers_file = os.path.expanduser(args.answers_file)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # data_loader = create_data_loader(questions, tokenizer, model.config)
    model: torch.nn.Module
    model.eval()
    sequence_bias = None
    def get_tokens_as_tuple(word):
        return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])

    max_new_tokens = args.max_tokens
    
    dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    new_dataset = []
    for example in tqdm(dataset):

        cur_prompt = example['instruction']
        

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        
        stop_str = conv_templates[args.conv_mode].sep2 if conv_templates[args.conv_mode].sep_style != SeparatorStyle.LLAMA_3 else conv_templates[args.conv_mode].stop_str
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if args.conv_mode == 'llama3':
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            terminators = tokenizer.eos_token_id
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True if args.temperature > 0 else False,
                attention_mask=attention_mask,
                temperature=args.temperature,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                output_attentions=model.config.output_attentions,
                sequence_bias=sequence_bias)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        # print(outputs)
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        new_example = deepcopy(example)
        new_example['output'] = outputs
        new_dataset.append(new_example)


    client.write(new_dataset, answers_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-molora", action="store_true")
    parser.add_argument("--unload-lora", action="store_true")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--s3-answers-file", type=str, default="s3://syj_test/answer.jsonl")
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-logit-bias", action='store_true')
    parser.add_argument("--logit-score", default=15.0)
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--ada-output-molora", action="store_true")
    parser.add_argument("--add-layer-index", type=int, default=0)
    parser.add_argument("--infer-answer", action="store_true")
    parser.add_argument("--unload-ffn", action="store_true")
    args = parser.parse_args()

    eval_model(args)