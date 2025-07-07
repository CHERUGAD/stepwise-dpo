# 🧠 Stepwise DPO — Direct Preference Optimization with LLM-based Stepwise Rewards

This project implements a custom **StepwiseDPOTrainer** that uses **LLM-generated step-wise rewards** to fine-tune a small language model using the DPO (Direct Preference Optimization) method.

Inspired by the ["Let’s Verify Step by Step" (OpenAI)](https://arxiv.org/abs/2408.15240v1) paper.

---

## 🚀 Project Summary

| Component                    | Implementation                                     |
|-----------------------------|----------------------------------------------------|
| Base Model                  | `google/flan-t5-small` (≤ 1.3B parameters)         |
| Trainer                     | `StepwiseDPOTrainer` (subclass of `trl.DPOTrainer`) |
| Reward Model                | `LLMRewardModel` using `flan-t5-small`             |
| Dataset                     | Custom synthetic pairs (`synthetic_prm.json`)      |
| Training Framework          | Hugging Face Transformers + TRL + PyTorch          |
| Inference Outputs           | `inference_results.csv`, `inference_results.jsonl` |

---

## 🧩 Stepwise Reward Modeling

The **LLMRewardModel** (`src/reward_model/reward_model.py`) uses a T5 model to score each reasoning step individually. Text is split into logical steps (via `\n`) and each step is scored independently using the language model.

These scores are **aggregated** and used as the reward signal during DPO training.

---

## 🧪 Trainer Customization

The `StepwiseDPOTrainer`:
- Subclasses Hugging Face’s `DPOTrainer`
- Overrides `compute_loss()` to:
  - Compute full-sequence log-likelihoods
  - Compute reward differences using step-wise scores from `LLMRewardModel`
  - Apply the final loss:  
    \[
    \mathcal{L} = -\log\sigma\left((r_c - r_r) \cdot (\log p_c - \log p_r)\right)
    \]

File: `src/trainer/stepwise_dpo_trainer.py`

---

## 🏋️ Training Instructions

```bash
# Convert raw dataset to DPO format
python scripts/convert_to_dpo_format.py

# Train the Stepwise DPO model
python scripts/train_stepwise_dpo.py

🔧 Training Config
Epochs: 3

Batch Size: 2

Learning Rate: 5e-5

Device: CPU

Output Directory: ./outputs/final_stepwise_dpo_model

📈 Inference & Evaluation
# Run batch inference with reward scoring
python scripts/inference_stepwise_dpo.py

🔍 Output Files
outputs/inference_results.jsonl

outputs/inference_results.csv

Each row contains:

json
Copy
Edit
{
  "prompt": "...",
  "model_output": "...",
  "chosen_score": float,
  "rejected_score": float,
  "preferred": "chosen" | "rejected"
}

📂 Project Structure (Trimmed)
├── data/
│   ├── raw/synthetic_prm.json
│   └── dpo/...
├── outputs/
│   ├── final_stepwise_dpo_model/
│   ├── inference_results.csv
├── scripts/
│   ├── train_stepwise_dpo.py
│   ├── inference_stepwise_dpo.py
│   └── convert_to_dpo_format.py
├── src/
│   ├── reward_model/reward_model.py
│   └── trainer/stepwise_dpo_trainer.py
├── LLM_USAGE.md
├── README.md


🧠 LLM Usage & Attribution
Developed with the assistance of ChatGPT (GPT-4). Logged in LLM_USAGE.md.

