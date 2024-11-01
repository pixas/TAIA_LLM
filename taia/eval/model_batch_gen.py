import argparse
import torch
import os
import json
from tqdm import tqdm, trange
# import shortuuid
from evalplus.data import get_human_eval_plus
from evalplus.data import get_mbpp_plus

from taia.conversations import conv_templates, SeparatorStyle
from taia.model.builder import load_pretrained_model, load_molora_pretrained_model, load_automerge_model
from taia.utils import disable_torch_init, get_model_name_from_path
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from taia.utils import client
from copy import deepcopy
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
# from PIL import Image
import math


    
class LogitBiasProcess(LogitsProcessor):
    def __init__(self, activate_token_list: list[int] = None, activate_scale=100):
        self.activate_token_list = activate_token_list
        self.activate_scale=activate_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # logger.info(scores.shape, input_ids.shape)
        for id_ in self.activate_token_list:
            if scores.dim() == 2:
                scores[:, id_] += self.activate_scale
            else:
                scores[id_] += self.activate_scale
        return scores

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_loss(logits, labels, attention_mask, vocab_size):
    from torch.nn import CrossEntropyLoss
    labels = labels.masked_fill(~attention_mask, -100)
    shift_logits = logits[..., :-1, :].contiguous()
    B, N, C = shift_logits.shape
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    # this loss is [-1, ], we need to reshape it to [B, N]
    loss = loss.reshape(B, N)
    # we must know that some positions are 0-loss because of ignore_index, we need to ignore these
    loss_sum = loss.sum(dim=-1)
    loss_actual_position = torch.not_equal(loss, 0).sum(dim=-1)
    loss = loss_sum / loss_actual_position  # [B, ]
    return loss


def generate_func(model, input_ids, **kwargs):
    if input_ids.dim() == 1:
        # only one item
        input_ids = input_ids.unsqueeze(0)
    max_new_tokens = kwargs.pop("max_new_tokens", args.max_new_tokens)
    tokenizer = kwargs.pop("tokenizer")
    sequence_bias = kwargs.pop("sequence_bias", None)
    if args.conv_mode == 'llama3':
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = tokenizer.eos_token_id
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            sequence_bias=sequence_bias,
            use_cache=True)
    return output_ids


# Custom dataset class
class CustomDataset:
    def __init__(self, questions, batch_size, conv_mode, task_specific_prompt, dataset_name='default', tokenizer=None,):
        self.questions = questions
        self.batch_size = batch_size
        self.size = len(questions)
        self.conv = conv_templates[conv_mode].copy()
        self.conv_mode = conv_mode
        self.task_specific_prompt = task_specific_prompt
        self.dataset_name = dataset_name
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        bz = self.batch_size

        # return question, ansewr, additional info
        questions = []
        prompts = []
        answers = []
        additional_infos = []
        for i in range(index*bz, (index+1)*bz):
            if i < self.size:
                conv = self.conv.copy()
                if self.dataset_name.endswith("plus"):
                    question = self.questions[i]['prompt']
                    questions.append(question)
                    if self.is_base: 
                        if self.few_shot_samples == "":
                            prompts.append(self.few_shot_samples + "Problem: " + question + "\n\nAnswer: The answer is ")
                        else:
                            prompts.append(self.few_shot_samples + question + "\n</problem>\n\n<AnswerText> Let's think step by step. ")
                    else:
                        conv.append_message(conv.roles[0], question+self.task_specific_prompt)
                        conv.append_message(conv.roles[1], None)
                        prompts.append(conv.get_prompt())
                    answers.append(None)
                    additional_infos.append(self.questions[i]['task_id'])
                else:
                    line = self.questions[i]
                    question = line['conversations'][0]['value']
                    questions.append(question)


                    question = question + self.task_specific_prompt
                    if self.conv_mode == 'qwen': 
                        
                        messages = deepcopy(self.messages)
                        
                        messages.append({"role": "user", "content": question})
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        prompts.append(text) 
                    else:
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompts.append(conv.get_prompt())
                    answers.append(line['conversations'][1]['value'] if len(line['conversations']) > 1 else None)
                    additional_infos.append(line['eval'] if 'eval' in line else None)

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return questions, prompts, answers, additional_infos

    def __len__(self):
        return len(self.questions) // self.batch_size + 1

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


    # pass

four_choices_datasets = ["logiqa_en", "mmedbench_en", "mmlu", "sat_math", "mmlu_math", "mmedbench_zh", "medmcqa", 'logiqa_en_prompt', "mmedbench_en_prompt", "mmlu_prompt", 'med_mmlu']
five_choices_datasets = ['commonsense_qa', "CMExam_zh", "medqa", 'commonsense_qa_prompt', "MedQA"]
three_choices_datasets = ['pubmedqa', 'pubmedqa_c']

def eval_model(args):
    # Model
    dataset_name = args.question_file.split("/")[-1].split(".")[0]
    test_base = model_path is None and args.model_base is not None 
    if args.question_file.split("/")[-1].split(".")[0] in ["mmedbench_zh", "ceval", "cmmlu", "race_high", "race_middle", "mmedbench_en", "mmlu", "arc", "winogrande"]:
        args.use_logit_bias = True
    

    if dataset_name == "mbpp_plus":
        
        questions = get_mbpp_plus()
        questions = [{"prompt": problem['prompt'], "task_id": task_id} for task_id, problem in questions.items()]
    elif dataset_name == "humaneval_plus":
        questions = get_human_eval_plus()
        # print(questions)
        questions = [{"prompt": problem['prompt'], "task_id": task_id} for task_id, problem in questions.items()]
    else:
        if args.question_file.endswith(".csv"):
            questions = pd.read_csv(args.question_file)
            questions = convert_to_json(questions)
        elif args.question_file.endswith(".jsonl"):
            questions = client.read_jsonl(os.path.expanduser(args.question_file))
        else:
            # a json file
            questions = client.read_json(os.path.expanduser(args.question_file))
    
    
    sequence_bias = None

    task_specific_prompt = ""
    
    if dataset_name == 'apps':
        task_specific_prompt = "\n\nPlease use python language to answer this problem. You should process stdin and stdout with input() and print():"

    elif dataset_name == 'bbh':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    elif dataset_name == 'gsm8k':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    elif dataset_name == 'math' or dataset_name == 'math_500':
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as:  The answer is {answer}."


    elif dataset_name in ["race_high", "race_middle", "mmedbench_en", "mmlu", "arc"]:

        task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."

    elif dataset_name in ["mmedbench_zh", "ceval", "cmmlu"]:

        task_specific_prompt = "\n\n请用选项的字母直接回答，不要输出其他信息："

    elif dataset_name == "humaneval":
        task_specific_prompt = "\n\nPlease complete the code within the code block ```python```."

    elif dataset_name == "logiqa_en":
        if test_base or args.conv_mode == 'qwen':
            
            task_specific_prompt = "\n\nPlease answer with option letter directly, do not output other infomation."
        else:
            task_specific_prompt = "\n\nPlease think step by step and give your answer in the end."
            
    elif dataset_name == "logiqa_zh":

        task_specific_prompt = "\n\n请用选项的字母直接回答，不要输出其他信息："

    elif dataset_name == "commonsense_qa":
        task_specific_prompt = "\n\nLet's think step by step. Please format the final answer at the end of the response as: The answer is {answer}."

    elif dataset_name == "svamp":
        task_specific_prompt = "\n\nPlease format the final answer at the end of the response as: The answer is {answer}."

    else:
        raise NotImplementedError
   
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # import pdb
    # pdb.set_trace()

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    if args.resume and os.path.exists(answers_file):
        data = client.read_jsonl(answers_file)
        current_file_num = len(data)

        questions = questions[current_file_num:]
        ans_file = open(answers_file, "a", encoding='utf-8')
    else:
        ans_file = open(answers_file, "w", encoding='utf-8')

    # print(tokenizer.pad_token, tokenizer.eos_token)
    if len(questions) == 0:
        exit(0)
    else:
        
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        if args.load_molora:
        # if "molora" in model_path:
            tokenizer, model, context_len, tokenizer_with_prefix_space = load_molora_pretrained_model(model_path, args.model_base, model_name, args.load_molora, use_logit_bias=args.use_logit_bias, add_layer_index=args.add_layer_index, ada_output_molora=args.ada_output_molora, unload_lora=args.unload_lora)
        else:
            tokenizer, model, context_len, tokenizer_with_prefix_space, base_model = load_pretrained_model(model_path, args.model_base, model_name, use_logit_bias=args.use_logit_bias, unload_ffn=args.unload_ffn, return_base=args.use_loracl)
        tokenizer.padding_side = "left"
        tokenizer_with_prefix_space.padding_side = "left"
        model: torch.nn.Module
        model.eval()
        if "32b" in model_path.lower() or (args.model_base is not None and "32b" in args.model_base.lower()):
            args.batch_size = 4
        if "truthfulqa_mc1" in dataset_name:
            args.batch_size = 1
        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
            args.conv_mode = args.conv_mode + '_mmtag'
            print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
        dataset = CustomDataset(questions, batch_size=args.batch_size, conv_mode=args.conv_mode, task_specific_prompt=task_specific_prompt, dataset_name=dataset_name , tokenizer=tokenizer, is_base=args.is_base, add_few_shot=args.add_few_shot, num_samples=args.fewshot_samples)

    for idx in trange(len(dataset)):
        questions, prompts, answers, additional_infos = dataset[idx]
        if len(questions) == 0:
            break

        input_ids = tokenizer(prompts, return_tensors='pt', padding=True).input_ids
        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
        if args.conv_mode == 'chatglm3' or args.conv_mode == 'chatglm2':
            attention_mask = attention_mask.to(torch.float)

        if args.conv_mode == 'llama3':
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            terminators = tokenizer.eos_token_id
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True if args.temperature > 0 else False,
                attention_mask=attention_mask,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                sequence_bias=sequence_bias,
                use_cache=True)
        # print(input_ids.shape, output_ids.shape)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # print("original outputs: ",tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True) )
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

        if args.infer_answer:
            if "zh" in prompts[0]:
                cot_prompt = "\n答案为"
            elif dataset_name in ["CMExam_cot", "CMB_cot", "cmmlu_cot", "ceval_cot", "medqa_mainland_cot"]:
                cot_prompt = "\n答案为"
            else:
                cot_prompt = "\nThe answer is "
            conv = conv_templates[args.conv_mode].copy()
            # cut_length = len(conv.sep2)
            cot_prompts = [(prompt + output + f"{' ' if output.strip().endswith('.') else '. '}{cot_prompt}") for prompt, output in zip(prompts, outputs)]
            input_ids = tokenizer(cot_prompts, return_tensors='pt', padding=True).input_ids.to(device='cuda', non_blocking=True)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device='cuda', non_blocking=True)
            if args.conv_mode == 'chatglm3' or args.conv_mode == 'chatglm2':
                attention_mask = attention_mask.to(torch.float)

            def add_choice_words(choices_word):
                choice_ids = [tokenizer.encode(x)[0] for x in choices_word]
                return choice_ids
            choices_word = None
            if dataset_name in four_choices_datasets:
                choices_word = ["A", "B", "C", "D"]

            elif dataset_name in five_choices_datasets:
                choices_word = ["A", "B", "C", "D", "E"]

            elif dataset_name in three_choices_datasets:
                choices_word = ['A', 'B', 'C']
            else:
                raise NotImplementedError
            
            if choices_word is not None:
                logits_processor = LogitsProcessorList()
                logits_processor.append(LogitBiasProcess(add_choice_words(choices_word), activate_scale=args.logit_score))
                cot_max_new_tokens = 1
            else:
                logits_processor = None
                cot_max_new_tokens = 50

            with torch.inference_mode():
                answer_output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=cot_max_new_tokens,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=logits_processor,
                    use_cache=True
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != answer_output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
           
            answer_outputs = tokenizer.batch_decode(answer_output_ids[:, input_token_len:], skip_special_tokens=True)
            # print(answer_outputs)
            outputs = [f"{output}{' ' if output.strip().endswith('.') else '. '}{cot_prompt}{answer_output}" for output, answer_output in zip(outputs, answer_outputs)]
            
        if dataset_name == 'humaneval_plus' or dataset_name == 'mbpp_plus':
            for question, output, answer, additional_info in zip(questions, outputs, answers, additional_infos):
                ans_file.write(json.dumps({
                    "task_id": additional_info,
                    "solution": output
                }) + "\n")
        else:
            for question, output, answer, additional_info in zip(questions, outputs, answers, additional_infos):
                ans_file.write(json.dumps({"prompt": question,
                                        "text": output,
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
    parser.add_argument("--is_base", action="store_true")
    parser.add_argument("--add_few_shot", action="store_true")
    parser.add_argument("--fewshot_samples", type=int, default=1)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")


    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-logit-bias", action='store_true')
    parser.add_argument("--logit-score", default=100.0)
    parser.add_argument("--conv-mode", type=str, default="qwen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--ada-output-molora", action="store_true")
    parser.add_argument("--add-layer-index", type=int, default=0)
    parser.add_argument("--infer-answer", action="store_true")
    parser.add_argument("--only-load", type=str, default=None)
    parser.add_argument("--use-loracl", action="store_true")
    parser.add_argument('--batch-size', type=int, default=8)
    
    
    

    args = parser.parse_args()

    eval_model(args)