# TAIA

This is the repository for "TAIA: Large Language Models are Out-of-distribution Data Learners"
<p align = "center">
<img src="img/taia.png" width="95%" alt="" align=center />
</p>
<p align = "center">
The comparison of TAIA with vanilla fine-tuning method on OOD data.
</p>

## Highlights
* Our proposed **TAIA** as an simple-yet-effective method which can be applied in OOD fine-tuning with large language models. It can avoid the disruption of FFN memories while acquiring superior downstream improvements. ![](img/cmp_other.png)
* **TAIA** can learn OOD features without disturbing internal knowledge. Therefore, it supports vertical task improvement via general fine-tuning data. ![](img/ood_tuning.png)
* **TAIA** can learn OOD features without disturbing internal knowledge. Therefore, it resists red-team attacks to a promising extent. ![](img/jailbreak.png)
* Our study achieves superb *data efficiency* (less data with higher performance), *domain generalization* (reservation of few-shot adaptation), and *high representation ability* (higher representation rank). ![](img/highlight.png)



## Setups

### Install

```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Data Preparation
You can download the `datas.tar.gz` from https://huggingface.co/datasets/pixas/TAIA_data.
Unzip the `datas.tar.gz` and put it as the `datas/` folder.

### Finetuning
#### LoRA Finetuning
```bash
sbatch scripts/train/slurm/sft_qwen1.5-1.8b-lora.sh
```

#### MoLoRA Finetuning

```bash
sbatch scripts/train/slurm/sft_qwen1.5-1.8b-molora.sh
```

#### Safety Finetuning
For the safety tuning, please first prepare the `llama2_7b_chat` model and check the checkpoint folder and `llama2` folder are specified in the following scripts properly.
For explicitly harmful attack, please run 
```bash
sbatch scripts/train/slurm/sft_llama2-lora-redteam.sh
```
For identity shifting attack, please run 
```bash
sbatch scripts/train/slurm/sft_llama2-lora-aoa.sh
```
For benign attack, please run
```bash
sbatch scripts/train/slurm/sft_llama2-lora-benign.sh
```




### Evaluation
The main evaluation is conducted on seven test sets, covering reasoning, math and knowledge understanding.
All the placeholder of the script **must be** replaced by your own paths 
For evaluation of TAIA, please first replace the domain in `scripts/eval/slurm/eval_lora_woffn.sh` as the domain you want to test, and run
```bash
bash scripts/eval/slurm/eval_lora_woffn.sh
```

For the safety evaluation, please run
```bash
bash scripts/eval/slurm/eval_advbench_woffn.sh
```

### Scaling of TAIA
Just replace the data path from `alpaca_gpt4_bilingual` to `mcot_1000`,`mcot_10000`,`mcot_50000`,`mcot_100k`,`mcot_01` and `mcot` and run the `lora` fine-tuning script with the `llama2` backbone.
The data sizes are changed from `[1k, 10k, 50k, 100k, 180k, 2.0m]`, separately.
The evaluation scripts are the same with `bash scripts/eval/slurm/eval_lora_woffn.sh`.
