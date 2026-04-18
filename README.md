# 🏆 Plan-RewardBench

<p align="center">
  <em>A Comprehensive Benchmark for Trajectory-Level Reward Modeling in Tool-Augmented Agents</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.08178"><img src="https://img.shields.io/badge/arXiv-2604.08178-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/wyy1112/Plan-RewardBench"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg?style=flat-square" alt="HuggingFace"></a>
  <a href="https://github.com/wyy-1112/Plan-RewardBench"><img src="https://img.shields.io/badge/GitHub-Code-181717.svg?style=flat-square&logo=github" alt="GitHub"></a>
  <a href="https://github.com/wyy-1112/Plan-RewardBench/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code-Apache_2.0-green.svg?style=flat-square" alt="License"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/Data-CC_BY_4.0-lightgrey.svg?style=flat-square" alt="Data License"></a>
  <img src="https://img.shields.io/badge/Pairs-1%2C171-blue.svg?style=flat-square" alt="Pairs">
  <img src="https://img.shields.io/badge/Scenarios-4_Families-orange.svg?style=flat-square" alt="Scenarios">
  <img src="https://img.shields.io/badge/Conference-ACL_2026_Main-purple.svg?style=flat-square" alt="ACL 2026 Main">
</p>

---

## 📖 Overview

**Plan-RewardBench** is a trajectory-level preference benchmark designed to evaluate how well reward models and LLM judges can distinguish high-quality agent trajectories from plausible near-misses in complex, multi-turn, tool-integrated reasoning scenarios.

Unlike existing benchmarks that focus on single-turn or atomic tool-call correctness, Plan-RewardBench targets **trajectory-level evaluation** — assessing entire interaction sequences including planning logic, tool usage patterns, error recovery, and safety adherence.

### Key Features

- **1,171 pairwise trajectory comparisons** across 7 evaluation splits
- **4 scenario families** covering diverse agentic capabilities
- **Hard-negative pairing** via minimal-edit perturbations for fine-grained discrimination
- **Unified pairwise protocol** with A/B swap for bias control
- **Three evaluator types**: Discriminative RMs, Generative RMs, and LLM-as-Judge

---

## 🗂️ Benchmark Structure

### Scenario Families

| Scenario Family | Splits | #Pairs | Description |
|---|---|---|---|
| **Complex Planning** | `planning_single_easy` | 144 | Single-turn planning with straightforward constraints |
| | `planning_single_hard` | 158 | Single-turn planning with complex/dynamic constraints |
| | `planning_multi_easy` | 109 | Multi-turn planning with moderate horizon |
| | `planning_multi_hard` | 73 | Multi-turn planning with long horizon (up to 64 turns) |
| **Robust Error Recovery** | `robust_recovery` | 361 | Recovery from tool errors, empty results, partial failures |
| **Safety Refusal** | `safety_refusal` | 51 | Distinguishing safe refusal from unsafe compliance |
| **Tool Irrelevance** | `tool_irrelevance` | 275 | Recognizing when tools are irrelevant or unavailable |

### Data Format

Each instance is a JSON object with the following structure:

```json
{
  "query": "User's task description",
  "tools": [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}],
  "uuid": "unique-identifier",
  "chosen": {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]},
  "reject": {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
}
```

Trajectories contain interleaved messages with roles: `user`, `assistant`, `tool_call`, and `tool_response`.

### Loading the Data

**From HuggingFace:**

```python
from datasets import load_dataset

dataset = load_dataset("wyy1112/Plan-RewardBench")

# Iterate over instances
for item in dataset["train"]:
    print(item["uuid"], item["_lcp_bucket"])
    chosen_msgs = item["chosen"]["messages"]
    reject_msgs = item["reject"]["messages"]
```

> **Note**: HuggingFace displays a single `train` split — this is simply the container for the full evaluation benchmark, **not** a training set. Plan-RewardBench is an **evaluation-only** benchmark.

**From local JSONL files:**

```python
import json

with open("benchmark/planning_multi_easy.jsonl") as f:
    for line in f:
        item = json.loads(line)
        print(item["uuid"], len(item["chosen"]["messages"]), "turns")
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/wyy-1112/Plan-RewardBench.git
cd Plan-RewardBench

# Create and activate conda environment
conda create -n plan-rewardbench python=3.10 -y
conda activate plan-rewardbench

pip install -r eval/requirements.txt
```

### Running Evaluation

**1. Set up API keys:**

```bash
export OPENAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
export QWEN_API_KEY="your-key"
export QWEN_BASE_URL="your-endpoint"
```

**2. Run with an LLM-as-Judge:**

```bash
# Evaluate a single model (must match a name in config.yaml)
python eval/evaluate_benchmark_final.py \
  --config eval/config.yaml \
  --data-dir benchmark \
  --output-dir results \
  --models gpt-4o \
  --workers 16 \
  --benchmarks all

# Evaluate multiple models at once
python eval/evaluate_benchmark_final.py \
  --config eval/config.yaml \
  --data-dir benchmark \
  --output-dir results \
  --models deepseek-r1 qwen-max \
  --workers 16 \
  --benchmarks all
```

**3. Run with a local Discriminative RM (requires GPU + torch):**

```bash
# Install PyTorch first if not already installed:
#   pip install torch  (see https://pytorch.org for CUDA-specific instructions)

python eval/evaluate_benchmark_final.py \
  --config eval/config.yaml \
  --data-dir benchmark \
  --output-dir results \
  --models skywork-reward-v2-llama3.1-8b \
  --workers 1 \
  --benchmarks all
```

**4. Run with a Generative RM (via vLLM):**

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Reward-Reasoning/RRM-32B \
  --served-model-name RRM-32B \
  --tensor-parallel-size 2 \
  --port 8000

# Run evaluation (model name must match config.yaml)
python eval/evaluate_benchmark_final.py \
  --config eval/config.yaml \
  --data-dir benchmark \
  --output-dir results \
  --models RRM-32B \
  --workers 8 \
  --benchmarks all
```

> **Note**: The `--models` argument must match a model `name` defined in [`eval/config.yaml`](eval/config.yaml). If no match is found, the script will report `"No matching models found"`.

See [`eval/run_eval.sh`](eval/run_eval.sh) for more examples.

### Configuration

Edit [`eval/config.yaml`](eval/config.yaml) to add your models. Three model types are supported:

| Type | `model_type` | Description |
|---|---|---|
| LLM-as-Judge | `llm_judge` | Any OpenAI-compatible API (GPT, DeepSeek, Qwen, local vLLM) |
| Discriminative RM | `bt` | Local Bradley-Terry reward model (HuggingFace, requires GPU) |
| Remote RM | `bt_remote` | Remote reward model via HTTP API |

---

## 📊 Main Results

Pairwise accuracy (%) on Plan-RewardBench. **Avg** is the macro-average across all 7 splits.

| Model | Type | Multi-E | Multi-H | Sngl-E | Sngl-H | Robust | Safety | Irrel. | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-Plus | LLM Judge | 68.35 | **68.77** | **84.55** | 74.68 | 73.75 | 55.88 | 63.73 | **69.96** |
| DeepSeek-V3.2-Exp | LLM Judge | 69.27 | 61.58 | 79.51 | **74.84** | 66.76 | 75.00 | 60.00 | 69.57 |
| Inf-ORM-Llama3.1-70B | Scalar RM | 70.31 | 65.03 | 79.86 | 74.05 | 69.78 | 58.53 | 66.91 | 69.21 |
| GPT-5 | LLM Judge | 63.99 | 45.82 | 83.85 | 62.18 | 69.39 | **84.80** | 69.73 | 68.54 |
| Gemini-3-Flash | LLM Judge | 66.36 | 47.53 | 81.08 | 67.25 | 67.31 | 78.43 | **75.55** | 69.07 |
| RRM-32B | Generative RM | 68.45 | 62.10 | 75.22 | 70.80 | 67.15 | 60.30 | 61.15 | 66.45 |
| Skywork-Reward-V2 (8B) | Scalar RM | 73.85 | 61.44 | 69.79 | 72.15 | 65.10 | 53.92 | 64.91 | 65.88 |

See the paper for full results across 21 models.

---

## 🔑 Key Findings

1. **No single evaluator dominates all categories.** The best overall model (Qwen-Plus, 69.96%) is not the best on Safety (GPT-5, 84.80%) or Tool-Irrelevance (Gemini-3-Flash, 75.55%).

2. **Multi-turn Hard planning remains the most challenging split**, with even the strongest models struggling to exceed 70%.

3. **Scalar RMs are competitive on explicit signals** (e.g., error recovery) but lack the reasoning depth for implicit planning logic.

4. **Context collapse occurs beyond 32K tokens**, where several evaluators fall below random chance.

---

## 📁 Repository Structure

```
Plan-RewardBench/
├── README.md
├── LICENSE                          # Apache 2.0 (code) + CC BY 4.0 (data)
├── .gitignore
├── benchmark/                       # Benchmark data (1,171 pairwise instances)
│   ├── planning_multi_easy.jsonl    # 109 pairs
│   ├── planning_multi_hard.jsonl    # 73 pairs
│   ├── planning_single_easy.jsonl   # 144 pairs
│   ├── planning_single_hard.jsonl   # 158 pairs
│   ├── robust_recovery.jsonl        # 361 pairs
│   ├── safety_refusal.jsonl         # 51 pairs
│   └── tool_irrelevance.jsonl       # 275 pairs
└── eval/                            # Evaluation scripts
    ├── evaluate_benchmark_final.py  # Main evaluation script
    ├── config.yaml                  # Model configuration
    ├── run_eval.sh                  # Example run commands
    └── requirements.txt             # Python dependencies
```

---

## 📜 License

- **Code** (evaluation scripts, configs): [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Data** (benchmark datasets): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## 📚 Citation

If you find Plan-RewardBench useful, please cite our paper:

```bibtex
@article{wang2026aligning,
  title={Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling},
  author={Wang, Jiaxuan and Hu, Yulan and Yang, Wenjin and Pan, Zheng and Li, Xin and Guo, Lan-Zhe},
  journal={arXiv preprint arXiv:2604.08178},
  year={2026}
}
```

---

## 📬 Contact

Questions or suggestions? Feel free to reach out:

- 📧 **Email**: jiaxuanwang@smail.nju.edu.cn
- 💬 **GitHub Issues**: [Open an issue](https://github.com/wyy-1112/Plan-RewardBench/issues)
