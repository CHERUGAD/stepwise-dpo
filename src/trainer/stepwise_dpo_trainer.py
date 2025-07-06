from trl import DPOTrainer
from typing import Dict, Any
import torch

class StepwiseDPOTrainer(DPOTrainer):
    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int = None  # Needed to avoid Trainer crash
    ):
        """
        Stepwise DPO loss: scores steps inside 'chosen' and 'rejected' separately,
        aggregates reward per sample, and computes DPO loss.
        """

        # === 1. Extract chosen and rejected completions ===
        chosen = inputs["chosen"]    # List[str]
        rejected = inputs["rejected"]

        # === 2. Split by step – assume each step begins with "\nStep " or similar
        chosen_steps = [c.split("\nStep ") for c in chosen]
        rejected_steps = [r.split("\nStep ") for r in rejected]

        # === 3. Stepwise reward scores (currently dummy values)
        # ⛏️ Later we’ll use an actual reward model
        chosen_rewards = [torch.tensor([1.0] * len(steps)) for steps in chosen_steps]
        rejected_rewards = [torch.tensor([0.0] * len(steps)) for steps in rejected_steps]

        # === 4. Aggregate per-sample rewards
        chosen_scores = torch.stack([r.mean() for r in chosen_rewards])
        rejected_scores = torch.stack([r.mean() for r in rejected_rewards])

        # === 5. Compute DPO loss
        beta = getattr(self, "beta", 0.1)
        loss = -torch.nn.functional.logsigmoid(beta * (chosen_scores - rejected_scores)).mean()

        return (loss, None) if return_outputs else loss
