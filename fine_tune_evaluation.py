import torch
from peft import PeftModel
from slm_fine_tune import SLMTrainer, parse_action
from sklearn.metrics import accuracy_score
import argparse

def main():
    parser = argparse.ArgumentParser(description="GRPO LoRA fine-tuned evaluation")
    parser.add_argument("--huggingface_model_name", required=True, type=str)
    parser.add_argument("--huggingface_token", required=True, type=str)
    parser.add_argument("--train_data_file_path", required=True, type=str)
    parser.add_argument("--fine_tune_path", required=True, type=str)

    args = parser.parse_args()
    trainer = SLMTrainer(
        hf_model_name=args.huggingface_model_name,
        hf_token=args.huggingface_token,
        train_file_path=args.train_data_file_path,
        model_save_path=None,
    )
    trainer.create_dataset()
    dataset = trainer.dataset
    tokenizer = trainer.tokenizer
    ft = PeftModel.from_pretrained(trainer.model, args.fine_tune_path)
    ft.eval()

    def generate_test_response(message_list):
        prompt = tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(ft.device)

        with torch.no_grad():
            out = ft.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False
            )

        gen_tokens = out[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        return answer

    test_message_lists = [d['prompt'] for d in dataset['test']]
    gt_responses = [d['ground_truth'] for d in dataset['test']]
    test_results = [generate_test_response(t) for t in test_message_lists]
    parsed_results = [parse_action(t) for t in test_results]
    parsed_gt = [parse_action(g) for g in gt_responses]

    accuracy = accuracy_score(y_true=parsed_gt, y_pred=parsed_results)
    print(accuracy)


if __name__ == "__main__":
    main()
