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
        chosen_input_ids = inputs["chosen_input_ids"].cpu()
        rejected_input_ids = inputs["rejected_input_ids"].cpu()

        # Decode
        chosen_texts = self.reward_model.tokenizer.batch_decode(
            chosen_input_ids, skip_special_tokens=True
        )
        rejected_texts = self.reward_model.tokenizer.batch_decode(
            rejected_input_ids, skip_special_tokens=True
        )

        def split_steps(text):
            return [s.strip() for s in text.split("\n") if s.strip()]

        # Compute stepwise scores
        chosen_scores = [
            sum(self.reward_model.score(split_steps(text))) for text in chosen_texts
        ]
        rejected_scores = [
            sum(self.reward_model.score(split_steps(text))) for text in rejected_texts
        ]

        chosen_rewards = torch.tensor(chosen_scores, dtype=torch.float32)
        rejected_rewards = torch.tensor(rejected_scores, dtype=torch.float32)

        # Ensure both are on same device
        chosen_rewards = chosen_rewards.to(model.device)
        rejected_rewards = rejected_rewards.to(model.device)

        # DPO loss
        preference_scores = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(preference_scores).mean()

        return (loss, None) if return_outputs else loss
