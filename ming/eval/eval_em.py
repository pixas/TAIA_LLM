import os 
import json 

import argparse 
from tqdm import tqdm, trange
from subprocess import PIPE, Popen, TimeoutExpired
import tempfile
import re 
from pathlib import Path

from sympy import sympify
try:
    from rouge import Rouge
except ImportError:
    import subprocess
    import sys
    
    # 使用 subprocess 执行 pip 安装
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge"])
    
    # 重新尝试导入
    try:
        from rouge import Rouge
    except ImportError:
        print("please install rouge manually")
        sys.exit(1)


def normalize_frac(x):
    # Pattern to match \frac{a}{b}
    pattern = r'\\frac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize_dfrac(x):
    pattern = r'\\dfrac\{([^\}]+)\}\{([^\}]+)\}'
    
    # Search for the pattern in the input string
    match = re.search(pattern, x)
    
    # If a match is found, extract 'a' and 'b'
    if match:
        a = match.group(1)  # Numerator
        b = match.group(2)  # Denominator
        
        # Convert to a simplified form, if necessary
        # For demonstration, just return the extracted parts
        return a, b
    else:
        # import pdb 
        # pdb.set_trace()
        return None

def normalize(x):
    if "\\frac" in x and normalize_frac(x):
        a, b = normalize_frac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
        
    elif "\\dfrac" in x and normalize_dfrac(x):
        a, b = normalize_dfrac(x)
        try:
            a = float(a)
            b = float(b)
            return a / b
        except:
            return x
    else:
        try:
            x = sympify(x).evalf()
            return float(x)
        except:
            return x

def acc(pred, target):
    return 1 if pred == target else 0

def rouge(pred, target):
    # compute rouge-1, rouge-2, rouge-l
    pass

def extract_bbox_content(s):
    contents = []
    i = 0
    while i < len(s):
        if s[i:i+7] == '\\boxed{':
            depth = 1
            start = i + 7
            i += 7
            while i < len(s) and depth > 0:
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    depth -= 1
                    if depth == 0:
                        contents.append(s[start:i])
                i += 1
        else:
            i += 1
    return contents


def extract_answer_content(s):
    match1 = re.search(r'the answer is (.*?)\.', s, )
    match2 = re.search(r'The answer is (.*?)\.', s, )

        
    if match1 is None:
        return [match2.group(1)] if match2 else [None]
    if match2 is None:
        return [match1.group(1)] if match1 else [None] 
    return [match1.group(1), match2.group(1)]
    # return match.group(1) if match else None


def math_acc(line):
    pred = line['text']
    target = line['additional_info']['solution']

    target_answer = extract_bbox_content(target)[0]
    pred_answer = extract_bbox_content(pred)
    if "\\text{" in target_answer:
        # remove \\text{ and the last }
        # target_answer  
        text_index = target_answer.index("\\text{")
        target_answer = target_answer[text_index + 6: -1]

    # print(target)
    # print(target_answer)
    # print(pred)

    if pred_answer != []:
        pred_answer = pred_answer[0]
        target_answer = normalize(target_answer)
        if isinstance(target_answer, float):
            pred_answer = normalize(pred_answer) #if pred_answer is not None else float("-inf")

        if isinstance(target_answer, str) and isinstance(pred_answer, str): # target type = str
            return 1.0 if target_answer in pred_answer else 0.0
        
        elif isinstance(pred_answer, str): # target type = float
            return 1.0 if pred_answer in target else 0.0
        
        elif isinstance(pred_answer, float):
            if abs(pred_answer - target_answer) < 1e-3:
                return 1.0
            else:
                return 0.0

        return 0
    else:
        if "the answer is" in pred or "The answer is":
            pred_answer = extract_answer_content(pred)
            return 1.0 if any(target_answer in str(x) for x in pred_answer) else 0.0
        else:
            pred_answer = pred[len(pred) // 2:]
            return 1.0 if target_answer in pred_answer else 0.0



def code_acc(line):
    cwd = os.getcwd()
    text = line['text']


    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    
    # 如果找到匹配项，则提取并打印
    if match:
        extracted_content = match.group(1)
    else:
        extracted_content = text

    additional_info = line['additional_info']
    # function_name = additional_info['function_name']
    test = additional_info['test']
    executable_code = extracted_content
    if isinstance(test, str):
        test_code = executable_code + "\n" + test
    else:
        test_code = executable_code + "\n" + "\n".join(test)
    
    if "def " not in test_code:
        test_code = additional_info.get("function_name", "") + test_code 
        
    if additional_info.get("entry_point", None) is not None:
        test_code = test_code + "\n\n" + f"check({additional_info['entry_point']})"
    

    with tempfile.TemporaryDirectory() as tempdir_name:
        tempdir_name = Path(tempdir_name)
        with open(tempdir_name / "program.py", "w", encoding="UTF-8") as f:
            f.write(test_code)
        os.chdir(tempdir_name)
        

    # idx = additional_info["id"]
    # with open(f"/remote-home/syjiang/repo/MING-MOE/logs/diverse/humaneval/tmp/{idx}", 'w') as f:
    #     f.write(test_code)
        
        p = Popen(f'python program.py', shell=True, stdout=PIPE, stderr=PIPE)
        time_limit = 15  # seconds
        scores = 1
        try:
            stdout, stderr = p.communicate(timeout=time_limit)
        except TimeoutExpired:
            # Linux
            # os.killpg(p.pid, signal.SIGTERM)
            # Windows
            os.system("kill {pid}".format(pid=p.pid))
            scores = 0
        else:
            if stderr:
                scores = 0
    

    os.chdir(cwd)
    return scores

def gsm8k_acc(line):
    # extract answer after #### 
    pred = line['text']
    target = line['additional_info']['answer']

    index = target.find("####")
    target_answer = target[index+4:].strip()
    

    pred_answer = extract_answer_content(pred)
    # import pdb
    # pdb.set_trace()
    # if index != -1:
    #     pred_answer = pred[index + 4:].strip()  # Extract answer after "####" and strip any leading or trailing whitespace
    # else:
    #     pred_answer = pred
    # index = target.find("####")
    # target_answer = target[index + 4:].strip()
    if pred_answer is not None:
        return 1 if any(target_answer in str(x) for x in  pred_answer) else 0
    else:
        return 0

def gsmic_acc(line):
    # extract answer after #### 
    pred = line['text']
    target = line['additional_info']['answer']


    target_answer = target
    

    pred_answer = extract_answer_content(pred)
    # import pdb
    # pdb.set_trace()
    # if index != -1:
    #     pred_answer = pred[index + 4:].strip()  # Extract answer after "####" and strip any leading or trailing whitespace
    # else:
    #     pred_answer = pred
    # index = target.find("####")
    # target_answer = target[index + 4:].strip()
    if pred_answer is not None:
        return 1 if any(target_answer in str(x) for x in pred_answer) else 0
    else:
        return 0

def sum_acc(line):
    pred = line['text']
    target = line['additional_info']['answer']
    rouge = Rouge()
    rouge_score = rouge.get_scores(pred, target)
    rouge_1 = rouge_score[0]['rouge-1']['r']
    rouge_2 = rouge_score[0]['rouge-2']['r']
    rouge_l = rouge_score[0]['rouge-l']['r']
    return rouge_1, rouge_2, rouge_l
    

def mmedbench_acc(line):
    pred = line['text']
    pred = re.findall(r'[A-E]', pred)[0]

    answer = line['additional_info']['answer_idx']

    return 1 if pred == answer else 0 

def bbh_acc(line):
    pred = line['text']
    answer = line['additional_info']['target']
    if "(" in pred and ")" in pred:
        # extract the content in () [maybe many], and select the one which is a single capital letter
        pred = re.findall(r'\((.*?)\)', pred)
        for p in pred:
            if p in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                pred = f"({p})"
                break


    return 1 if answer in pred else 0

def apps_acc(line):
    text = line['text']
    match = re.search(r"```python(.*?)```", text)

    if match:
        extracted_content = match.group(1)
    else:
        extracted_content = text
    additional_info = line['additional_info']
    input_output = additional_info['input_output']
    # try:
    #     input_output = json.loads(input_output)
    # except:
    #     return None
    # input_output = json.loads(input_output)

    inputs = input_output['inputs']
    outputs = input_output['outputs']
    test_code = extracted_content 
    assert len(inputs) == len(outputs)
    
    ff = tempfile.NamedTemporaryFile(mode='w')
    ff.write(test_code)
    name = ff.name 
    scores = 1
    for i in range(len(inputs)):
        cur_input = inputs[i]
        cur_output = outputs[i]
        
        p = Popen(f'python {name} < {cur_input}', shell=True, stdout=PIPE, stderr=PIPE)
        time_limit = 15  # seconds
        try:
            stdout, stderr = p.communicate(timeout=time_limit)
        except TimeoutExpired:
            # Linux
            # os.killpg(p.pid, signal.SIGTERM)
            # Windows
            # Popen("TASKKILL /F /PID {pid} /T".format(pid=p.pid))
            scores = 0
            break
        if stderr:
            scores = 0
            break
        if stdout.strip() != cur_output.strip():
            scores = 0
            break
    ff.close()
    return scores

def triviaqa_acc(line):
    pred = line['text']
    answers = line['additional_info']['answer']
    for answer in answers:
        if pred == answer:
            return 1 
    return 0


def mc_acc(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if pred.endswith("."):
        pred = pred[:-1]
    return 1 if pred == answer else 0

winogrande = mmlu_acc = arc_acc = cmmlu_acc = ceval_acc = mc_acc

def commonsense_qa_acc(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if len(pred) == 1:
        return 1 if pred == answer else 0
    extract_pred = extract_answer_content(pred)
    if extract_pred is not None:
        return 1 if any(answer in str(x) for x in extract_pred) else 0
    else:
        return 0

    
def logiqa_en(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if len(pred) == 1:
        return 1 if pred == answer else 0
    extract_pred = extract_answer_content(pred)
    if extract_pred is not None:
        if "(" in extract_pred:
            extract_pred = extract_pred[extract_pred.index("(") + 1: extract_pred.index(")")]
    else:
        return 0
    return 1 if any(answer in x for x in extract_pred) else 0

def logiqa_zh(line):
    pred = line['text']
    answer = line['additional_info']['answer']
    if len(pred) == 1:
        return 1 if pred == answer else 0
    match = re.search(r'答案是 (.*?)\。', pred, re.IGNORECASE)

    extract_pred = match.group(1) if match else None
    # extract_pred = extract_answer_content(pred)
    if extract_pred is not None:
        if "(" in extract_pred:
            extract_pred = extract_pred[extract_pred.index("(") + 1: extract_pred.index(")")]
    return 1 if answer == extract_pred else 0

def svamp_acc(line):
    pred = line['text']
    target = line['additional_info']['answer']

    target_answer = target
    pred_answer = extract_bbox_content(pred)

    # print(target)
    # print(target_answer)
    # print(pred)

    if pred_answer != []:
        pred_answer = pred_answer[0]
        if isinstance(target_answer, float):
            pred_answer = normalize(pred_answer) #if pred_answer is not None else float("-inf")

        if isinstance(target_answer, str) and isinstance(pred_answer, str): # target type = str
            return 1.0 if target_answer in pred_answer else 0.0
        
        elif isinstance(pred_answer, str): # target type = float
            return 1.0 if pred_answer in str(target) else 0.0
        
        elif isinstance(pred_answer, float):
            if abs(pred_answer - target_answer) < 1e-3:
                return 1.0
            else:
                return 0.0

        return 0
    else:
        if "the answer is" in pred or "The answer is":
            pred_answer = extract_answer_content(pred)
            try:
                return 1.0 if any(abs(target_answer - float(x)) < 1e-3 for x in pred_answer) else 0.0
            except:
                if abs(target_answer - int(target_answer)) < 1e-4:
                    # the answer can be expressed as int 
                    target_answer = str(int(target_answer))
                    return 1.0 if any(target_answer in str(x) for x in pred_answer) else 0.0
                else:
                    # the answer is float and the pred_answer cannot be converted to float 
                    return 0
        else:
            pred_answer = pred[len(pred) // 2:]
            return 1.0 if target_answer in pred_answer else 0.0


METRIC_FUNC_MAPPING = {
    "math": math_acc,
    "math_500": math_acc,
    "humaneval": code_acc,
    "mbpp": code_acc,
    "gsm8k": gsm8k_acc,
    "mmedbench_en": mmedbench_acc,
    "mmedbench_zh": mmedbench_acc,
    "bbh": bbh_acc,
    "apps": apps_acc,
    "triviaqa": triviaqa_acc,
    "winogrande": winogrande,
    "mmlu": mmlu_acc,
    "arc": arc_acc,
    "cmmlu": cmmlu_acc,
    "ceval": ceval_acc,
    "GSM-IC_mstep_new": gsmic_acc,
    "GSM-IC_2step_new": gsmic_acc,
    "commonsense_qa": commonsense_qa_acc,
    "svamp": svamp_acc,
    "logiqa_en": logiqa_en,
    "logiqa_zh": logiqa_zh,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False)
    args = parser.parse_args()

    # input_file is a jsonl file with the following format:
    # questions = client.read_jsonl(args.input_file)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.input_file), "r")]
    
    total_num = len(questions)
    total_score = 0
    rouge_score = [[], [], []]

    if "merge" in args.input_file:
        dataset_name = args.input_file.split("/")[-3]
    else:
        dataset_name  = args.input_file.split("/")[-2].replace(".jsonl", "")
        if dataset_name not in METRIC_FUNC_MAPPING:
            dataset_name  = args.input_file.split("/")[-1].replace(".jsonl", "")
    acc_func = METRIC_FUNC_MAPPING[dataset_name]
    wrong_idx = []
    for line in tqdm(questions, total=total_num):
        scores = acc_func(line)
        if isinstance(scores, tuple):
            rouge_1, rouge_2, rouge_l = scores
            rouge_score[0].append(rouge_1)
            rouge_score[1].append(rouge_2)
            rouge_score[2].append(rouge_l)
            continue
        if scores is None:
            total_num -= 1
            wrong_idx.append(line)
            continue
        total_score += scores
        if scores == 0:
            wrong_idx.append(line)
    if rouge_score[0]:
        print(f"Rouge-1: {sum(rouge_score[0]) / len(rouge_score[0])}")
        print(f"Rouge-2: {sum(rouge_score[1]) / len(rouge_score[1])}")
        print(f"Rouge-l: {sum(rouge_score[2]) / len(rouge_score[2])}")
        exit(0)
    avg_acc = total_score / total_num
    print(f"Acc in {dataset_name}: {avg_acc}")
    # if args.output_file:
    #     client.write_json(wrong_idx, args.output_file, ensure_ascii=False, indent=4)
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(wrong_idx, f, ensure_ascii=False)