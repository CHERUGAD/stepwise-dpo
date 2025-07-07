
import sys
import os
sys.path.append(os.path.abspath("."))  
import json
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.reward_model.reward_model import LLMRewardModel
from tqdm import tqdm

# Load model and tokenizer
model_path = "./outputs/final_stepwise_dpo_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
reward_model = LLMRewardModel(model_name="google/flan-t5-small", device="cpu")

# Load synthetic prompts
with open("data/raw/synthetic_prm.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

def split_steps(text):
    return [s.strip() for s in text.split("\n") if s.strip()]

print("\n🔄 Running Batch Evaluation...\n")

for entry in tqdm(data):
    prompt = entry["prompt"]
    chosen = entry["chosen"]
    rejected = entry["rejected"]

    # Model-generated answer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_new_tokens=64)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Stepwise scoring
    score_chosen = sum(reward_model.score(split_steps(chosen)))
    score_rejected = sum(reward_model.score(split_steps(rejected)))
    preferred = "chosen" if score_chosen > score_rejected else "rejected"

    results.append({
        "prompt": prompt,
        "model_output": generated,
        "chosen_score": score_chosen,
        "rejected_score": score_rejected,
        "preferred": preferred
    })

# Save as JSONL
jsonl_path = "outputs/inference_results.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Also save as CSV
csv_path = "outputs/inference_results.csv"
with open(csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Batch evaluation completed.")
print(f"📁 Results saved to:\n - {jsonl_path}\n - {csv_path}")
