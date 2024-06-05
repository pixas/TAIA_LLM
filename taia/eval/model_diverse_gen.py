import argparse
import torch
import os
import json
from tqdm import trange
from taia.conversations import conv_templates, SeparatorStyle
from taia.model.builder import load_pretrained_model, load_molora_pretrained_model
from taia.utils import get_model_name_from_path

from torch.utils.data import DataLoader
import pandas as pd 
from torch.nn import CrossEntropyLoss

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


def get_loss(logits, labels, vocab_size):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def lora_cl(base_model, loaded_model, input_ids: torch.Tensor):
    # calculate the loss of input_ids on base_model
    with torch.inference_mode():
        base_logits = base_model(input_ids).logits
        base_loss = get_loss(base_logits, input_ids, base_model.config.vocab_size)
        # calculate the loss of input_ids on loaded_model
        loaded_logits = loaded_model(input_ids).logits
        loaded_loss = get_loss(loaded_logits, input_ids, loaded_model.config.vocab_size)
    print(base_loss, loaded_loss)
    if base_loss < loaded_loss:
        return base_model 
    else:
        return loaded_model

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
        tokenizer, model, context_len, tokenizer_with_prefix_space, base_model = load_pretrained_model(model_path, args.model_base, model_name, use_logit_bias=args.use_logit_bias, unload_ffn=args.unload_ffn, return_base=args.use_loracl)

    # load args.question_file, which is a csv file
    if args.question_file.endswith(".csv"):
        questions = pd.read_csv(args.question_file)
        questions = convert_to_json(questions)
    elif args.question_file.endswith(".jsonl"):
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    else:
        with open(args.question_file, 'r') as f:
            questions = json.load(f)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    
    if args.resume:
        current_file_num = 0
        if os.path.exists(answers_file):
        # if os.path.exists(answers_file):
            with open(answers_file, 'r') as f:
                for line in f:
                    current_file_num += 1
            questions = questions[current_file_num:]
            ans_file = open(answers_file, "a", encoding='utf-8')
        else:
            ans_file = open(answers_file, "w", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')

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
    task_specific_prompt = ""
    dataset_name = args.question_file.split("/")[-1].split(".")[0]

    if dataset_name == 'apps':
        task_specific_prompt = "\n\nPlease use python language to answer this problem. You should process stdin and stdout with input() and print():"

    elif dataset_name == 'bbh':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    elif dataset_name == 'gsm8k':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    elif dataset_name == 'math' or dataset_name == 'math_500':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as:  The answer is {answer}."

    elif dataset_name in ["winogrande"]:
        sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B"]}
        max_new_tokens = 2 if test_base else 1 
        if test_base and args.conv_mode == 'llama3':
            max_new_tokens = 3
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."

    elif dataset_name in ["race_high", "race_middle", "mmedbench_en", "mmlu", "arc"]:
        sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
        max_new_tokens = 2 if test_base else 1 
        if test_base and args.conv_mode == 'llama3':
            max_new_tokens = 3
        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."

    elif dataset_name in ["mmedbench_zh", "ceval", "cmmlu"]:
        sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
        max_new_tokens = 2 if test_base else 1 
        if test_base and args.conv_mode == 'llama3':
            max_new_tokens = 3
        task_specific_prompt = "\n\n请用选项的字母直接回答，不要输出其他信息："

    elif dataset_name == "humaneval":
        task_specific_prompt = "\n\nPlease complete the code within the code block ```python```."

    elif dataset_name == "logiqa_en":
        if test_base or args.conv_mode == 'qwen':
            sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
            max_new_tokens = 1 if test_base else 1 
            task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
        else:
            task_specific_prompt = "\n\nPlease think step by step and give your answer in the end."
            
    elif dataset_name == "logiqa_zh":
        sequence_bias = {get_tokens_as_tuple(x): args.logit_score for x in ["A", "B", "C", "D"]}
        max_new_tokens = 1 if args.conv_mode != "llama2" else 1 
        task_specific_prompt = "\n\n请用选项的字母直接回答，不要输出其他信息："

    elif dataset_name == "commonsense_qa":
        task_specific_prompt = "\n\nLet's think step by step. Please format the final answer at the end of the response as: The answer is {answer}."

    elif dataset_name == "svamp":
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    else:
        raise NotImplementedError
        
    dataset = CustomDataset(questions)
    if args.use_loracl:
        base_model = base_model.to(torch.float16)
        print("LoRA Continual Learning begins..")
    for idx in trange(len(dataset)):
        line = dataset[idx]
        question, answer, additional_info = line
        question = question + task_specific_prompt
        cur_prompt = question 
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids

        stop_str = conv_templates[args.conv_mode].sep2 if conv_templates[args.conv_mode].sep_style != SeparatorStyle.LLAMA_3 else conv_templates[args.conv_mode].stop_str
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if args.use_loracl:
            
            model = lora_cl(base_model, model, input_ids)

        
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
        if dataset_name == 'bbh':
            pass
        else:
            if args.infer_answer and (dataset_name in ['svamp', 'math', 'math_500', 'commonsense_qa']):
                
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], cur_prompt)
                add_char = " " if outputs.endswith(".") else ". "
                conv.append_message(conv.roles[1], outputs + f"{add_char}The answer is ")
                prompt = conv.get_prompt()
                begin_index = prompt.index(outputs + f"{add_char}The answer is ")
                end_index = begin_index + len(outputs + f"{add_char}The answer is ")
                prompt = prompt[:end_index]

                # print(prompt)
                if dataset_name == "commonsense_qa":
                    cur_sequence_bias = {get_tokens_as_tuple(x): 100.0 for x in ["A", "B", "C", "D", "E"]}
                    new_max_new_tokens = 2 if test_base else 1 
                    if test_base and args.conv_mode == 'llama3':
                        new_max_new_tokens = 1
                elif dataset_name == "logiqa_en" or dataset_name == 'mmlu' or dataset_name == 'mmedbench_en' or dataset_name ==  'mmedbench_zh':
                    cur_sequence_bias = {get_tokens_as_tuple(x): 100.0 for x in ["A", "B", "C", "D"]}
                    new_max_new_tokens = 2 if test_base else 1 
                    if test_base and args.conv_mode == 'llama3':
                        new_max_new_tokens = 1
                else:
                    cur_sequence_bias = None
                    new_max_new_tokens = 10

                input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device='cuda', non_blocking=True)
                attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.pad_token_id,
                        attention_mask=attention_mask,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=new_max_new_tokens,
                        use_cache=True,
                        output_attentions=model.config.output_attentions,
                        sequence_bias=cur_sequence_bias)
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                answer = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                answer = answer.strip()
                if answer.endswith(stop_str):
                    answer = answer[:-len(stop_str)]
                answer = answer.strip()
                
                if dataset_name == 'mmlu' or dataset_name == 'mmedbench_en' or dataset_name == 'mmedbench_zh':
                    outputs = answer 
                else:
                    if not answer.endswith("."):
                        answer += "."
                    outputs = outputs + f"{add_char}The answer is {answer}"

        ans_file.write(json.dumps({"prompt": cur_prompt,
                                   "text": outputs,
                                   "solution": answer,
                                   "additional_info": additional_info,
                                   "model_id": model_name,
                                   "metadata": {}}, ensure_ascii=False) + "\n",)
        ans_file.flush()

        
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-molora", action="store_true")
    parser.add_argument("--unload-lora", action="store_true")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
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
    parser.add_argument("--use-loracl", action="store_true")
    args = parser.parse_args()

    eval_model(args)