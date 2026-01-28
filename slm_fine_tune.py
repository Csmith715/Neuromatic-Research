import pandas as pd
import re
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import login
import argparse

def preprocess_function(example):
    system_label = (
        "You are a lead qualification assistant.\n"
        "You will be given call transcript excerpts (tool output).\n"
        "Return exactly ONE Action command of the form:\n"
        "<respond> LABELS </respond>\n"
        "where LABELS is either:\n"
        "- None\n"
        "- or a comma-separated subset of: Authority, Budget, Timeline, Need\n"
        "No other text."
    )
    return {
        "prompt": [
            # {"role": "system", "content": example["instruction"]},
            {"role": "system", "content": system_label},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
            {"role": "tool", "content": example["query_results"]},
            {"role": "user", "content": (
                "Now provide the final qualification result.\n"
                "Return exactly one action command:\n"
                "<respond> LABELS </respond>\n"
                "LABELS must be None or a comma-separated subset of: Authority, Budget, Timeline, Need."
            )},
        ],
        "ground_truth": example["answers"]
    }

# For Gemma
# def preprocess_function(example):
#     return {
#         "prompt": [
#             {
#                 "role": "user",
#                 "content": f"{LABEL_SYS}\n\n{example['input']}"
#             },
#             {
#                 "role": "assistant",
#                 "content": example["output"]
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     f"Tool Results: {example['query_results']}\n\n"
#                     "Now provide the final qualification result.\n"
#                     "Return exactly one action command:\n"
#                     "<respond> LABELS </respond>\n"
#                     "LABELS must be None or a comma-separated subset of: Authority, Budget, Timeline, Need."
#                 )
#             },
#         ],
#         "ground_truth": example["answers"]
#     }


ALLOWED = {"Authority", "Budget", "Timeline", "Need"}
CANON_ORDER = ["Authority", "Budget", "Timeline", "Need"]

RESP_RE = re.compile(r"^\s*<\s*respond\s*>\s*(.*?)\s*<\s*/\s*respond\s*>\s*$", re.I | re.S)

def normalize_answer(text: str) -> str:
    t = text.strip()
    t = re.sub(r'^[`"\']+|[`"\']+$', "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t

def parse_label_set(t: str):
    t = normalize_answer(t)

    # None is valid and exclusive
    if t.lower() == "none":
        return "None"

    parts = [p.strip() for p in t.split(",") if p.strip()]
    if not parts:
        return "None"

    if any(p not in ALLOWED for p in parts):
        return None

    uniq = sorted(set(parts), key=lambda x: CANON_ORDER.index(x))
    return ", ".join(uniq)

# def parse_action(text: str):
#     m = RESP_RE.search(text.strip())
#     if not m:
#         return None
#     inner = m.group(1)
#     return parse_label_set(inner)

def parse_action(text: str):
    """
    Enforce <respond> ... </respond>. Return canonical label string or None if invalid.
    """
    m = RESP_RE.match(text.strip())
    if not m:
        return None
    inner = m.group(1)
    return parse_label_set(inner)

def reward_fn(completions, **kwargs):
    texts = []
    for c in completions:
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list) and len(c) and isinstance(c[-1], dict) and "content" in c[-1]:
            texts.append(c[-1]["content"])
        else:
            texts.append(str(c))

    golds = kwargs.get("ground_truth", None)

    rewards = []
    for i, out in enumerate(texts):
        pred = parse_action(out)
        if pred is None:
            rewards.append(-2.0)
            continue
        r = 0.5
        if golds is not None:
            gold = parse_action(golds[i])
            if gold is not None and pred == gold:
                r = 2.0
        else:
            r = 1.0

        rewards.append(r)

    return rewards


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

class SLMTrainer:
    def __init__(self, hf_model_name, hf_token, train_file_path, model_save_path):
        login(hf_token)
        self.model_name = hf_model_name
        self.training_file_path = train_file_path
        self.save_path = model_save_path
        self.dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            token=hf_token,
            use_fast=True
        )
        # If tokenizer doesn't have pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            token=hf_token,
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            quantization_config=quant_config,
        )
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.config = GRPOConfig(
            output_dir="grpo-lora",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            max_prompt_length=2048,  # Prompt length is another area to potentially focus on
            max_completion_length=32,
            num_generations=6,
            generation_batch_size=6,
            temperature=0.9,  # I need to adjust future trainings with a lower temp to nudge towards more strict generations
            bf16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=100,
            report_to=[]
        )

    def slm_training_pipeline(self):
        self.create_dataset()
        self.train_model()
        self.evaluate_model()

    def create_dataset(self):
        df = pd.read_csv(self.training_file_path)
        df = df.fillna('')

        new_answers = [f"<respond> {a} </respond>" if a else f"<respond> None </respond>" for a in df['answers']]
        df['answers'] = new_answers

        df_train = df.sample(frac=0.9, random_state=42)
        df_test = df.drop(df_train.index).reset_index(drop=True)
        df_train = df_train.reset_index(drop=True)

        hf_dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "test": Dataset.from_pandas(df_test)
        })

        self.dataset = hf_dataset_dict.map(preprocess_function, remove_columns=["instruction", "input", "output", "query_results", "answers"])

    def train_model(self):
        trainer = GRPOTrainer(
            model=self.model,
            args=self.config,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            reward_funcs=reward_fn,
            peft_config=self.peft_config,
            processing_class=self.tokenizer
        )

        trainer.train()
        trainer.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)

    def evaluate_model(self):
        base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            load_in_4bit=True,
        )

        ft = PeftModel.from_pretrained(base, self.save_path)
        ft.eval()

def main():
    parser = argparse.ArgumentParser(description="GRPO LoRA fine-tuning runner")
    parser.add_argument("--huggingface_model_name", required=True, type=str)
    parser.add_argument("--huggingface_token", required=True, type=str)
    parser.add_argument("--train_data_file_path", required=True, type=str)
    parser.add_argument("--model_save_path", required=True, type=str)

    args = parser.parse_args()

    trainer = SLMTrainer(
        hf_model_name=args.huggingface_model_name,
        hf_token=args.huggingface_token,
        train_file_path=args.train_data_file_path,
        model_save_path=args.model_save_path,
    )
    trainer.slm_training_pipeline()


if __name__ == "__main__":
    main()
