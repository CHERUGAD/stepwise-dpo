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

