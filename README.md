# Language Enhanced Model for Eye (LEME): An Open-Source Ophthalmology-Specific Large Language Model

Explore our [LEME](https://huggingface.co/hippocrates/leme_70b) (**L**anguage **E**nhanced **M**odel for **E**ye) model on Hugging Face! LEME was trained using the [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b-hf) framework and curated over 110,000 instructions from 19 subtasks, sourced from a vast pool of over 130,000 domainophthalmology-specific related ophthalmology data sources, including literature, patient case reports, and educational materials. You can also access the [instruction fine-tuning datasets and internal test dataset](https://huggingface.co/datasets/hippocrates/Ophtho_Instruction_FT_train) directly on Hugging Face.

## Running the prediction script for GPT models

```bash
python run_gpt.py \
 --file_path {dataset/test/independent/LongFormQA.parquet} \
 --model {gpt-3.5-turbo-0613 | gpt-4-0613 } \
```

## Evaluation

Before evaluation, please clone the [BARTScore repository](https://github.com/neulab/BARTScore) into your current directory and download the `bart_score.pth` model file from [BART checkpoint](https://drive.google.com/u/0/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m&export=download) for BART metrics evaluation. 

For abstract completion, fill-in-blank, short-answer QA and long-form QA tasks, please use `eval_generative.py` to obatin Rouge-1, Rouge-2, Rouge-L, Bert scores and BART scores.

```bash
python run_eval_generative.py \
 --file_path {path_to_prediction_file} \
```

For Multiple Choice Question (MCQ), please `run_eval_mcq.py` to obtain accuracy, micro-F1 and macro-F1 scores. Note you can customize the `extraction_function` within the script to extract A, B, C or D from raw outputs of various models.


```bash
python run_eval_mcq.py \
 --file_path {path_to_prediction_file} \
 --extraction_function=extract_first_letter
```

# Model Performance Comparison

## 1. Internal Validations

### 1.1 Abstract Completion

| Model         | Mean ± Std (95% CI)       | P-value (vs. LEME) |
|---------------|---------------------------|--------------------|
| GPT-3.5       | 0.17 ± 0.02 (0.14, 0.20)  | < 0.0001           |
| GPT-4         | 0.19 ± 0.02 (0.16, 0.2)   | 0.00178            |
| Llama 2 7B    | 0.17 ± 0.02 (0.13, 0.20)  | < 0.0001           |
| Llama 2 13B   | 0.16 ± 0.02 (0.12, 0.19)  | < 0.0001           |
| Llama 2 70B   | 0.10 ± 0.01 (0.08, 0.13)  | < 0.0001           |
| PMC-LLAMA 13B | 0.09 ± 0.01 (0.07, 0.12)  | < 0.0001           |
| Meditron 70B  | 0.07 ± 0.01 (0.06, 0.09)  | < 0.0001           |
| Eye-LLAMA     | 0.08 ± 0.01 (0.06, 0.09)  | < 0.0001           |
| LEME (Ours)   | 0.20 ± 0.03 (0.15, 0.25)  | -                  |

### 1.2 Fill-in-Blank

| Model         | Mean ± Std (95% CI)       | P-value (vs. LEME) |
|---------------|---------------------------|--------------------|
| GPT-3.5       | 0.72 ± 0.04 (0.64, 0.81)  | < 0.0001           |
| GPT-4         | 0.78 ± 0.03 (0.73, 0.85)  | 0.0140             |
| Llama 2 7B    | 0.34 ± 0.05 (0.26, 0.44)  | < 0.0001           |
| Llama 2 13B   | 0.37 ± 0.06 (0.27, 0.47)  | < 0.0001           |
| Llama 2 70B   | 0.16 ± 0.04 (0.10, 0.23)  | < 0.0001           |
| PMC-LLAMA 13B | 0.34 ± 0.05 (0.26, 0.46)  | < 0.0001           |
| Meditron 70B  | 0.19 ± 0.02 (0.14, 0.23)  | < 0.0001           |
| Eye-LLAMA     | 0.15 ± 0.01 (0.13, 0.18)  | < 0.0001           |
| LEME (Ours)   | 0.82 ± 0.04 (0.76, 0.89)  | -                  |

### 1.3 MCQ (Internal)

| Model         | Mean ± Std (95% CI)       | P-value (vs. LEME) |
|---------------|---------------------------|--------------------|
| GPT-3.5       | 0.53 ± 0.08 (0.37, 0.67)  | < 0.0001           |
| GPT-4         | 0.81 ± 0.07 (0.67, 0.93)  | < 0.0001           |
| Llama 2 7B    | 0.03 ± 0.03 (0.00, 0.10)  | < 0.0001           |
| Llama 2 13B   | 0.07 ± 0.05 (0.00, 0.20)  | < 0.0001           |
| Llama 2 70B   | 0.22 ± 0.07 (0.10, 0.37)  | < 0.0001           |
| PMC-LLAMA 13B | 0.04 ± 0.04 (0.00, 0.12)  | < 0.0001           |
| Meditron 70B  | 0.46 ± 0.09 (0.28, 0.60)  | < 0.0001           |
| Eye-LLAMA     | 0.24 ± 0.07 (0.12, 0.37)  | < 0.0001           |
| LEME (Ours)   | 0.57 ± 0.09 (0.42, 0.73)  | -                  |

### 1.4 Short-answer

| Model         | Mean ± Std (95% CI)       | P-value (vs. LEME) |
|---------------|---------------------------|--------------------|
| GPT-3.5       | 0.16 ± 0.03 (0.10, 0.23)  | < 0.0001           |
| GPT-4         | 0.20 ± 0.05 (0.11, 0.32)  | 0.0140             |
| Llama 2 7B    | 0.11 ± 0.03 (0.06, 0.18)  | < 0.0001           |
| Llama 2 13B   | 0.11 ± 0.03 (0.05, 0.17)  | < 0.0001           |
| Llama 2 70B   | 0.10 ± 0.03 (0.06, 0.18)  | < 0.0001           |
| PMC-LLAMA 13B | 0.08 ± 0.02 (0.04, 0.11)  | < 0.0001           |
| Meditron 70B  | 0.03 ± 0.00 (0.02, 0.04)  | < 0.0001           |
| Eye-LLAMA     | 0.02 ± 0.00 (0.01, 0.03)  | < 0.0001           |
| LEME (Ours)   | 0.22 ± 0.05 (0.18, 0.30)  | -                  |

## 2. External Validations

### 2.1 MCQ (External)

| Model         | Mean ± Std (95% CI)       | P-value (vs. LEME) |
|---------------|---------------------------|--------------------|
| GPT-3.5       | 0.45 ± 0.10 (0.27, 0.63)  | < 0.0001           |
| GPT-4         | 0.79 ± 0.07 (0.67, 0.90)  | < 0.0001           |
| Llama 2 7B    | 0.25 ± 0.08 (0.10, 0.40)  | < 0.0001           |
| Llama 2 13B   | 0.32 ± 0.09 (0.17, 0.47)  | < 0.0001           |
| Llama 2 70B   | 0.37 ± 0.08 (0.22, 0.52)  | < 0.0001           |
| PMC-LLAMA 13B | 0.14 ± 0.06 (0.07, 0.23)  | < 0.0001           |
| Meditron 70B  | 0.42 ± 0.10 (0.25, 0.60)  | < 0.0001           |
| Eye-LLAMA     | 0.29 ± 0.08 (0.13, 0.43)  | < 0.0001           |
| LEME (Ours)   | 0.68 ± 0.09 (0.53, 0.83)  | -                  |

### 2.2 Long-form

| Model         | Mean ± Std (95% CI)       | P-value (vs. LEME) |
|---------------|---------------------------|--------------------|
| GPT-3.5       | 0.17 ± 0.01 (0.16, 0.19)  | < 0.0001           |
| GPT-4         | 0.18 ± 0.01 (0.16, 0.19)  | < 0.0001           |
| Llama 2 7B    | 0.16 ± 0.01 (0.15, 0.18)  | < 0.0001           |
| Llama 2 13B   | 0.14 ± 0.01 (0.12, 0.15)  | < 0.0001           |
| Llama 2 70B   | 0.15 ± 0.01 (0.14, 0.16)  | < 0.0001           |
| PMC-LLAMA 13B | 0.16 ± 0.01 (0.14, 0.18)  | < 0.0001           |
| Meditron 70B  | 0.13 ± 0.01 (0.11, 0.15)  | < 0.0001           |
| Eye-LLAMA     | 0.13 ± 0.01 (0.11, 0.15)  | < 0.0001           |
| LEME (Ours)   | 0.19 ± 0.01 (0.17, 0.21)  | -                  |
