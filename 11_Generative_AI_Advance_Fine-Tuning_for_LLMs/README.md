# Generative AI Advanced Fine-Tuning for LLMs – Project Implementations

This folder contains advanced LLM alignment and fine-tuning experiments developed during the  
**IBM AI Engineering Professional Certificate**.

The focus of this module was aligning language models with human preferences using 
reward modeling, reinforcement learning from human feedback (RLHF), and 
direct preference optimization (DPO).

---

## 🧠 Overview

Key areas explored:

- Instruction tuning and reward modeling
- Reinforcement Learning from Human Feedback (RLHF)
- Proximal Policy Optimization (PPO) for language models
- Direct Preference Optimization (DPO)
- Preference-based alignment workflows using Hugging Face TRL

These projects extend earlier transformer fine-tuning work toward 
behavioral alignment and policy optimization for generative AI systems.

---

## 📂 Selected Implementations

### 🔹 Reward Modeling

- `reward_modelling_gpt2.py`  
  Fine-tuned GPT-2 as a reward model using LoRA on the  
  **Dahoas/synthetic-instruct-gptj-pairwise** dataset.  
  Achieved **71% pairwise accuracy**.

---

### 🔹 RLHF with PPO

- `rlhf_ppo_sentiment.py`  
  Applied PPO-based RLHF to steer GPT-2 behavior toward 
  "Happy" vs "Pessimistic" conversational styles using IMDb data.  
  Achieved **85% sentiment alignment**.

---

### 🔹 Direct Preference Optimization (DPO)

- `dpo_llm_alignment.py`  
  Implemented DPO-based alignment using Hugging Face TRL, 
  including dataset preparation, training, and evaluation workflows.

<img src="Images/loss dpo.png" width="240"/>
<img src="Images/samples_dpo.png" width="520"/>

---

## 🔧 Tools & Libraries

Python • PyTorch • Hugging Face Transformers • TRL • NumPy • Matplotlib

---

## 📌 Context

This module represents the LLM alignment and reinforcement learning component 
of the IBM AI Engineering Professional Certificate, bridging transformer fine-tuning 
with reinforcement learning and production-oriented generative AI systems.
