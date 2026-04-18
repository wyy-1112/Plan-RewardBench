#!/bin/bash
# Plan-RewardBench Evaluation Script
# Usage examples for different model types

# =============================================
# 1. Set API keys (replace with your own)
# =============================================
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export QWEN_API_KEY="your-qwen-api-key"

# =============================================
# 2. Run evaluation
# =============================================

# --- LLM-as-Judge (API-based) ---
# Evaluate a single model on all benchmarks
python evaluate_benchmark_final.py \
  --config config.yaml \
  --data-dir ../benchmark \
  --output-dir ./results \
  --models gpt-4o \
  --workers 16 \
  --benchmarks all

# Evaluate multiple models
python evaluate_benchmark_final.py \
  --config config.yaml \
  --data-dir ../benchmark \
  --output-dir ./results \
  --models deepseek-r1 qwen-max \
  --workers 16 \
  --benchmarks all

# --- Generative Reward Models (local vLLM) ---
# First, start vLLM server:
#   python -m vllm.entrypoints.openai.api_server \
#     --model Reward-Reasoning/RRM-32B \
#     --served-model-name RRM-32B \
#     --tensor-parallel-size 2 \
#     --port 8000
#
# Then run evaluation:
python evaluate_benchmark_final.py \
  --config config.yaml \
  --data-dir ../benchmark \
  --output-dir ./results \
  --models RRM-32B \
  --workers 8 \
  --benchmarks all

# --- Discriminative Reward Models (local GPU) ---
# BT models run single-threaded (GPU inference)
python evaluate_benchmark_final.py \
  --config config.yaml \
  --data-dir ../benchmark \
  --output-dir ./results \
  --models skywork-reward-v2-llama3.1-8b \
  --workers 1 \
  --benchmarks all

# --- Run on specific benchmarks only ---
python evaluate_benchmark_final.py \
  --config config.yaml \
  --data-dir ../benchmark \
  --output-dir ./results \
  --models qwen-max \
  --workers 16 \
  --benchmarks planning_multi_easy planning_single_hard robust_recovery
