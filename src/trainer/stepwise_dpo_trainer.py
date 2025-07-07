import sys
import os
sys.path.append(os.path.abspath("."))

from trl import DPOTrainer
from typing import Dict, Any
from src.reward_model.reward_model import LLMRewardModel
import torch
import torch.nn.functional as F

class StepwiseDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_model = LLMRewardModel(model_name="google/flan-t5-small", device="cpu")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs
    ):
        # Extract inputs
        chosen_input_ids = inputs["chosen_input_ids"]
        rejected_input_ids = inputs["rejected_input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # Forward pass to get logits
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            decoder_input_ids=chosen_input_ids,
            attention_mask=attention_mask
        )
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            decoder_input_ids=rejected_input_ids,
            attention_mask=attention_mask
        )

        # Log-probabilities
        chosen_log_probs = torch.log_softmax(chosen_outputs.logits, dim=-1)
        rejected_log_probs = torch.log_softmax(rejected_outputs.logits, dim=-1)

        # Sequence logprobs: mean over sequence length
        chosen_loglikelihood = chosen_log_probs.gather(
            dim=-1, index=chosen_input_ids.unsqueeze(-1)
        ).squeeze(-1).mean(dim=-1)

        rejected_loglikelihood = rejected_log_probs.gather(
            dim=-1, index=rejected_input_ids.unsqueeze(-1)
        ).squeeze(-1).mean(dim=-1)

        # Reward scores from stepwise reward model (on CPU)
        chosen_texts = self.reward_model.tokenizer.batch_decode(
            chosen_input_ids.cpu(), skip_special_tokens=True
        )
        rejected_texts = self.reward_model.tokenizer.batch_decode(
            rejected_input_ids.cpu(), skip_special_tokens=True
        )

        def split_steps(text):
            return [s.strip() for s in text.split("\n") if s.strip()]

        chosen_rewards = torch.tensor([
            sum(self.reward_model.score(split_steps(text)))
            for text in chosen_texts
        ], dtype=torch.float32, device=chosen_input_ids.device)

        rejected_rewards = torch.tensor([
            sum(self.reward_model.score(split_steps(text)))
            for text in rejected_texts
        ], dtype=torch.float32, device=rejected_input_ids.device)

        # Stepwise preference signal (acts as label)
        preference = chosen_rewards - rejected_rewards

        # Final loss: push chosen_loglikelihood > rejected_loglikelihood
        logits_diff = chosen_loglikelihood - rejected_loglikelihood
        loss = -F.logsigmoid(preference * logits_diff).mean()

        return (loss, None) if return_outputs else loss
