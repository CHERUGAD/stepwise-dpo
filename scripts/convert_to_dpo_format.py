import json
from datasets import Dataset, DatasetDict
import os

def convert_raw_to_dpo(raw_path: str, output_dir: str):
    # Load raw JSON data
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Transform raw_data into DPO format entries
    dpo_data = []
    for entry in raw_data:
        # Example structure of each entry:
        # {
        #   "prompt": "...",
        #   "chosen": "...",
        #   "rejected": "..."
        # }

        # Adjust this part based on your actual raw data keys
        prompt = entry.get("prompt", "")
        chosen = entry.get("chosen", "")
        rejected = entry.get("rejected", "")

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
            # Add stepwise reward info here if you have it, e.g., "stepwise_rewards": [...]
        })

    # Convert to Huggingface Dataset
    dataset = Dataset.from_list(dpo_data)

    # Split dataset into train/test if needed, or save whole dataset
    dataset_dict = DatasetDict({"train": dataset})

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print(f"Saved DPO dataset to {output_dir}")

if __name__ == "__main__":
    convert_raw_to_dpo("data/raw/synthetic_prm.json", "data/dpo")
