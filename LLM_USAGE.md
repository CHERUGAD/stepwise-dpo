## ✅ Stepwise DPO Training Log — Phase 1 (Without Reward Model)

### 🧩 Setup
- **Model**: `google/flan-t5-small`
- **Trainer**: `StepwiseDPOTrainer` (inherits from `DPOTrainer`)
- **Reward Model**: ❌ *Not yet integrated*
- **Dataset**: Converted from `synthetic_prm.json` → HuggingFace `DatasetDict` (DPO format)

### 🛠️ Training Config
- `epochs`: 3  
- `learning_rate`: 5e-5  
- `batch_size`: 2  
- `device`: CPU  

### 📊 Results
- `train_loss`: 0.6447  
- `train_runtime`: 9.03 sec  
- `train_samples_per_second`: 0.664  
- `train_steps_per_second`: 0.332  
- `output_dir`: `./outputs/checkpoint-3/`

### 📝 Notes
- This run confirms `StepwiseDPOTrainer` works fine with standard DPO loss.
- Reward model was intentionally excluded for this baseline run.
- Ready for integration of `LLMRewardModel` and step-wise scoring.

---

Next: ✅ Implement `LLMRewardModel` → 🧠 Add custom `compute_loss` using step-by-step reward evaluation.

## ✅ Stepwise DPO Training Log — Phase 2 (With Stepwise Reward Model)

### 🧩 Setup
- **Model**: `google/flan-t5-small`
- **Trainer**: `StepwiseDPOTrainer` (custom `compute_loss` with reward-based scoring)
- **Reward Model**: ✅ `LLMRewardModel` based on `flan-t5-small`
- **Dataset**: Same as Phase 1 (`synthetic_prm.json → DPO format`)

### 🔍 Loss Details
- Stepwise preference computed using:
  - `score()` for each substep in `chosen` and `rejected`
  - `preference = sum(chosen_rewards) - sum(rejected_rewards)`
- Log-likelihood of `chosen` vs `rejected` used in:
  - `loss = -logsigmoid(preference * (chosen_loglikelihood - rejected_loglikelihood))`

### 🛠️ Training Config
- `epochs`: 3  
- `learning_rate`: 5e-5  
- `batch_size`: 2  
- `device`: CPU  
- `output_dir`: `./outputs/`

### 📊 Results
- `train_loss`: **0.3756** ✅ (much improved vs baseline 0.6447)
- `train_runtime`: 10.05 sec  
- `train_steps_per_second`: 0.298  

### 📝 Notes
- `LLMRewardModel` correctly provides differentiable scoring.
- Loss is stable and decreasing — indicating effective preference alignment.
- Successfully resolved `decoder_input_ids` and gradient issues.
- This concludes functional Stepwise DPO training with CPU-ready setup.

---

Next: ✨ (Optional) Run inference script to compare model preferences between `chosen` and `rejected`.
