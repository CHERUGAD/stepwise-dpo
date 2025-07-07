import sys
import os
sys.path.append(os.path.abspath("."))  # enable local src imports

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import DatasetDict
from src.trainer.stepwise_dpo_trainer import StepwiseDPOTrainer
from trl import DPOConfig
import torch

model_name = "google/flan-t5-small"
dataset = DatasetDict.load_from_disk("data/dpo")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}  # or use `"auto"` if running on GPU
)

dpo_config = DPOConfig(
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    output_dir="./outputs",
    remove_unused_columns=False,
    report_to=None,
    bf16=False,
    fp16=False,
    auto_find_batch_size=True,
)

trainer = StepwiseDPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset["train"]
)

trainer.train()
