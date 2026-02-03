# tele-lens

This repository is for the paper:

**[No Global Plan in Chain-of-Thought: Uncover the Latent Planning Horizon of LLMs](https://arxiv.org/abs/2602.02103)**

In this paper, we uncovered a *myopic* latent planning horizon in LLMs' Chain-of-Thought (CoT), through our probing method **Tele-Lens**. We further underscore the significance of exploiting such CoT dynamics, with our proposed methods for estimation of both CoT uncertainty and necessity.

### Data

Our test set for probing is available at [Huggingface](https://huggingface.co/datasets/lxucs/tele-lens), spanning 12 tasks of diverse domains, which we categorize into three types.
- Explicit Compositional Tasks: tasks requiring explicit multi-step procedures to resolve, e.g. algorithmic reasoning
- Implicit Compositional Tasks: tasks requiring multiple reasoning steps but in a more nuanced and implicit manner, e.g. mathematical and logical reasoning
- Semantics & Knowledge Tasks: tasks focusing on semantic understanding and knowledge-based reasoning, e.g. MMLU

For each problem, a full response from both **Qwen3-32B** and our **In-Domain LLM** is also attached respectively.

### In-Domain LLM

We provide our In-Domain LLM, available for download at [Huggingface](https://huggingface.co/lxucs/tele-lens-llm).

This model is trained with GRPO upon [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), which learns task-aware reasoning behaviors. The resulting CoT trajectories are substantially shorter than those from Qwen3 models. 

In-Domain LLM should always be used with the following as **SYSTEM PROMPT**:

```text
You are a helpful assistant. Now the user asks you to solve a reasoning problem. You need to first think about the solving process in the mind and then provide the user with the answer. The thinking process is enclosed within <think> </think> tags, i.e., <think> thinking process here </think> final answer.
```

### Tele-Lens Adapter

The implementation for Tele-Lens adapter is provided at [adapter.py](adapter.py).

The adapter takes in hidden states and outputs the predicted logits on the LLM vocabulary (can be the whole vocabulary or a subset).

*Please feel free to open an issue for any questions regarding the probing details or results.*

### Paper Citation

```bibtex
@misc{xu2026globalplanchainofthoughtuncover,
      title={No Global Plan in Chain-of-Thought: Uncover the Latent Planning Horizon of LLMs}, 
      author={Liyan Xu and Mo Yu and Fandong Meng and Jie Zhou},
      year={2026},
      eprint={2602.02103},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02103}, 
}
```
