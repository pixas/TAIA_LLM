import os 
import json
import argparse 
from taia.utils import client
from taia.eval.eval_em import METRIC_FUNC_MAPPING
from typing import List, Any
from transformers import AutoTokenizer

def analysis(load_all_file: str, load_attn_file: str, metric: Any):
    load_all_data = client.read_jsonl(load_all_file)
    load_attn_data = client.read_jsonl(load_attn_file)
    assert len(load_all_data) == len(load_attn_data)
    all_correct_attn_wrong = []
    all_wrong_attn_correct = []
    all_correct_attn_correct = []
    # iterate over all data and find out the index where load_all is correct but load_attn is wrong
    for i in range(len(load_all_data)):
        load_all_line = load_all_data[i]
        load_attn_line = load_attn_data[i]
        
        load_all_acc = metric(load_all_line)
        load_attn_acc = metric(load_attn_line)
        
        if load_all_acc == 1 and load_attn_acc == 0:
            all_correct_attn_wrong.append(i)
        elif load_all_acc == 0 and load_attn_acc == 1:
            all_wrong_attn_correct.append(i)
        elif load_all_acc == 1 and load_attn_acc == 1:
            all_correct_attn_correct.append(i)
    return all_correct_attn_wrong, all_wrong_attn_correct, all_correct_attn_correct

def probe_entropy_with_correctness(load_all_entropy_file: str, load_attn_entropy_file: str, all_correct_attn_wrong: List[int]):
    load_all_entropy_data = client.read_csv(load_all_entropy_file)[1:-2]
    load_attn_entropy_data = client.read_csv(load_attn_entropy_file)[1:-2]
    
    assert len(load_all_entropy_data) == len(load_attn_entropy_data)
    all_correct_attn_wrong_entropy = []
    # all_wrong_attn_correct_entropy = []
    for i in all_correct_attn_wrong:
        load_all_entropy_line = load_all_entropy_data[i]
        load_attn_entropy_line = load_attn_entropy_data[i]
        
        load_all_entropy = [float(x) for x in load_all_entropy_line.split("\t")]
        load_attn_entropy = [float(x) for x in load_attn_entropy_line.split("\t")]
        if sum(load_all_entropy[12:]) > sum(load_attn_entropy[12:]) :
            # if the first layer of load-all is higher, then choose this result
            all_correct_attn_wrong_entropy.append(i)

    return all_correct_attn_wrong_entropy

def probe_length_with_correctness(load_all_file: str, load_attn_file: str, all_correct_attn_wrong: List[int]):
    load_all_data = client.read_jsonl(load_all_file)
    load_attn_data = client.read_jsonl(load_attn_file)
    all_correct_attn_wrong_entropy = []
    for i in all_correct_attn_wrong:
        load_all_line = load_all_data[i]
        load_attn_line = load_attn_data[i]
        if len(tokenizer.encode(load_all_line['text'])) > len(tokenizer.encode(load_attn_line['text'])):
            all_correct_attn_wrong_entropy.append(i)
    
    return all_correct_attn_wrong_entropy

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/models--Qwen--Qwen1.5-1.8B-Chat"))
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_all_file", type=str, help="the file path which is generated when ffn and attn are both loaded")
    parser.add_argument("--load_attn_file", type=str, help="the file path which is generated when only attn is loaded")
    parser.add_argument("--load_all_entropy_file", type=str, help="csv file; the first line and the last two lines are not used for analysis")
    parser.add_argument("--load_attn_entropy_file", type=str, help="csv file; the first line and the last two lines are not used for analysis")
    
    args = parser.parse_args()
    dataset = args.load_all_file.split("/")[-2]
    metric = METRIC_FUNC_MAPPING[dataset]
    
    all_correct_attn_wrong, all_wrong_attn_correct, all_correct_attn_correct = analysis(args.load_all_file, args.load_attn_file, metric=metric)
    print(f"all_correct_attn_wrong: {all_correct_attn_wrong}")
    print(f"all_wrong_attn_correct: {all_wrong_attn_correct}")
    print(f"all_correct_attn_correct: {all_correct_attn_correct}")
    print("All correct number:", len(all_correct_attn_wrong) + len(all_correct_attn_correct))
    print("Attn correct number:", len(all_correct_attn_correct) + len(all_wrong_attn_correct))
    # all_correct_attn_wrong_first_larger = probe_entropy_with_correctness(args.load_all_entropy_file, args.load_attn_entropy_file, all_correct_attn_wrong)
    all_correct_attn_wrong_first_larger = probe_length_with_correctness(args.load_all_file, args.load_attn_file, all_correct_attn_wrong)
    print(f"all_correct_attn_wrong_first_larger: {all_correct_attn_wrong_first_larger}")
    # all_wrong_attn_correct_first_larger = probe_entropy_with_correctness(args.load_attn_entropy_file, args.load_all_entropy_file, all_wrong_attn_correct)
    all_wrong_attn_correct_first_larger = probe_length_with_correctness(args.load_attn_file, args.load_all_file, all_wrong_attn_correct)
    print(f"all_wrong_attn_correct_first_larger: {all_wrong_attn_correct_first_larger}")
    print(len(all_correct_attn_correct) + len(all_wrong_attn_correct_first_larger) + len(all_correct_attn_wrong_first_larger))