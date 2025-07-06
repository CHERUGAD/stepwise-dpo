# src/reward_model/reward_model.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMRewardModel:
    def __init__(self, model_name: str = "microsoft/phi-2", device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def score(self, steps: list[str]) -> list[float]:
        """Returns a reward score per step."""
        scores = []
        for step in steps:
            inputs = self.tokenizer(step, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits[:, -1].mean().item()
                scores.append(score)
        return scores
