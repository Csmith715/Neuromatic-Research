# GRPO LoRA Fine-Tuning Pipeline

This repository contains a reusable Python training script for **fine-tuning small/medium language models using GRPO (Generalized Reward-Policy Optimization) with LoRA adapters**.

The pipeline is designed for **strict, structured outputs** (e.g. classification via XML-style tags) and is optimized for:

* Repetitive experimentation
* Low-VRAM training via 4-bit quantization
* Deterministic, reward-driven behavior

---

## Overview

The script trains an instruction-tuned causal language model to output **exactly one structured action** of the form:

```
<respond> LABELS </respond>
```

Where `LABELS` is either:

* `None`, or
* A comma-separated subset of: `Authority`, `Budget`, `Timeline`, `Need`

The model is trained using **TRL’s `GRPOTrainer`** with a custom reward function that:

* Strongly penalizes invalid formats
* Rewards valid structure
* Gives maximum reward for exact label-set matches

---

## Features

* ✅ GRPO + LoRA fine-tuning
* ✅ 4-bit quantized base model loading (GPU only)
* ✅ Canonical label normalization and validation
* ✅ Strict output-format enforcement
* ✅ CLI-driven training (no hard-coded paths)
* ✅ Train / test split with held-out evaluation

---

## Repository Structure

```
.
├── slm_fine_tune.py  
├── requirements.txt
├── README.md 
├── data/
   └── SLM_Data_LQ_Augmented_397.csv   # Training data

```

---

## Requirements

### Python

* Python 3.10+ recommended

### Core Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** CUDA-enabled GPU is strongly recommended. 4-bit loading is automatically disabled on CPU.

---

## Training Data Format

The training CSV must contain the following columns:

| Column Name     | Description                           |
|-----------------|---------------------------------------|
| `instruction`   | System prompt text                    |
| `input`         | User input/question/direction         |
| `output`        | Agent/LLM generated query             |
| `query_results` | Salesforce query response             |
| `answers`       | Gold labels (e.g. `Authority,Budget`) |

Example:

```csv
instruction,input,output,query_results,answers
"You are an expert in Salesforce and...", "Can this lead be qualified based on...","<execute> SELECT Id, Body__c, CreatedDate, LeadId__c FROM ...","Salesforce instance output: ...", "Authority"
```

During preprocessing, answers are automatically wrapped as:

```
<respond> Authority </respond>
```

The provided training data consists of all CRM-Arena Lead Qualification examples and 300 synthetically generated examples

---

## Usage

### Command Line Arguments

| Argument                   | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `--huggingface_model_name` | Base model ID (e.g. `Qwen/Qwen2.5-3B-Instruct`) |
| `--huggingface_token`      | Hugging Face access token                       |
| `--train_data_file_path`   | Path to training CSV                            |
| `--model_save_path`        | Output directory for LoRA adapters              |

### Example Run

```bash
python train_grpo.py \
  --huggingface_model_name "Qwen/Qwen2.5-3B-Instruct" \
  --huggingface_token "hf_XXXXXXXX" \
  --train_data_file_path "./data/SLM_Data_LQ_Augmented_397.csv" \
  --model_save_path "./out/qwen-grpo-lora"
```

---

## Training Details

### Optimization Strategy

* **Algorithm:** GRPO (policy optimization via reward functions)
* **Adapter:** LoRA (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
* **Precision:**

  * GPU: BF16 + 4-bit base model
  * CPU: FP32

### Key Hyperparameters

| Parameter              | Value             |
| ---------------------- | ----------------- |
| Learning Rate          | `2e-5`            |
| Batch Size             | `1` (accumulated) |
| Grad Accumulation      | `8`               |
| Generations per Prompt | `6`               |
| Temperature            | `0.6`             |
| Max Completion Length  | `32`              |

---

## Reward Function Behavior

The reward function assigns:

* **−2.0** → Invalid or malformed output
* **+0.5** → Valid `<respond>` structure
* **+2.0** → Exact label-set match with gold

Canonical ordering is enforced:

```
Authority, Budget, Timeline, Need
```

Duplicates or invalid labels are rejected.

---

## Evaluation

The current evaluation step:

* Loads the base model + trained LoRA adapters
* Sets the model to `eval()` mode

> ⚠️ For production use, you should add:

* Invalid-format rate
* Exact-match accuracy
* Confusion analysis per label

---


