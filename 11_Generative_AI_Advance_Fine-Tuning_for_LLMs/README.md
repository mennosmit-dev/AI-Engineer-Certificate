# Course 11: Generative AI Advanced Fine-Tuning for LLMs

This folder contains coursework and projects completed for the **[Generative AI Advanced Fine-Tuning for LLMs](https://www.coursera.org/learn/generative-ai-advanced-fine-tuning-for-llms?specialization=ai-engineer)** course, part of the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera.

## 🧠 Course Description

This course delves into advanced techniques for fine-tuning large language models (LLMs), focusing on aligning models with specific business needs through human feedback and preference optimization. Learners gain hands-on experience with instruction-tuning, reward modeling, and reinforcement learning methods using Hugging Face tools.

By the end of this course, you will be able to:

- Implement instruction-tuning and reward modeling using Hugging Face.
- Apply reinforcement learning from human feedback (RLHF) techniques, including proximal policy optimization (PPO) and direct preference optimization (DPO).
- Utilize LLMs as policies for generating responses based on input text.
- Calculate rewards using human feedback and train models to optimize performance.

---

## 📂 Contents: The coding projects I worked on

- `reward_modelling_gpt2.py`: Fine-tuned the GPT2 model to reward the quality of response of LLMs, which is later used for Reinforcement Learning with Human Feedback (RLHF) (trained on "Dahoas/synthetic-instruct-gptj-pairwise" using LoRa). The final pairwise accuracy of the model was 71%.
- `rlhf_ppo_sentiment.py`: Trained GPT-2 models with Reinforcement Learning from Human Feedback (RLHF) using PPO on the IMDb dataset to shape 'Happy' and 'Pessimistic' LLM behaviors for customer service scenarios. The models achieved 85% sentiment alignment with the target style.
- `dpo_llm_alignment.py`: Fine-tuned large language models using Direct Preference Optimization (DPO) with Hugging Face’s trl library to better align outputs with human preferences. Implemented dataset preparation, training, and evaluation to improve model performance in real-world NLP tasks. <br>
 <img src="Images/loss dpo.png" alt="train and validation loss of the dpo model" width="600"/> <img src="Images/samples_dpo.png" alt="sample dpo" width="200"/> 
---

## 🔧 Tools and Libraries

- Python
- Jupyter Notebooks
- PyTorch
- Hugging Face Transformers
- NumPy
- Matplotlib

---

## 📌 Certificate Series

This is the eleventh course in the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer).
