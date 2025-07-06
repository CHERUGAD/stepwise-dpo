# stepwise-dpo
Stepwise Direct Preference Optimization (DPO) implementation based on "Let's Verify Step by Step." Includes an LLM-based stepwise reward model, custom StepwiseDPOTrainer subclass, training on lightweight models like google/flan-t5-small, and reproducible scripts with experiment tracking.

## Project Structure

stepwise-dpo/
├── data/
│   ├── raw/                  # Raw dataset files, e.g., prm800k or generated data
│   └── processed/            # Processed datasets ready for training
│
├── src/
│   ├── reward_model/         # Code for building LLM-based reward model
│   ├── trainer/              # Custom StepwiseDPOTrainer subclass here
│   ├── utils/                # Utility functions/helpers
│   └── __init__.py
│
├── scripts/                  # Training, evaluation, data processing scripts
│   ├── convert_to_dpo_format.py
│   └── train_stepwise_dpo.py
│
├── experiments/              # Experiment logs, results, tensorboard runs
│
├── notebooks/                # Jupyter notebooks for exploratory work
│
├── outputs/                  # Model checkpoints, saved outputs
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and instructions
├── .gitignore
└── LLM_USAGE.md              # Track AI assistance / generated code notes

