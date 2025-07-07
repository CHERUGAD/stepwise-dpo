from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LLMRewardModel:
    def __init__(self, model_name: str = "google/flan-t5-small", device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device

    def score(self, steps: list[str]) -> list[float]:
        scores = []
        for step in steps:
            inputs = self.tokenizer(step, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, decoder_input_ids=inputs['input_ids'])
                reward = outputs.logits[:, -1].mean().item()
                scores.append(reward)
        return scores
