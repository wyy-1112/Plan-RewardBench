#!/usr/bin/env python3
"""
Plan-RewardBench Evaluation Script  
"""

import argparse
import json
import os
import time
import requests
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm import tqdm
import yaml
from dataclasses import dataclass
from collections import defaultdict

# ============================================================
# 1. Token Counter
# ============================================================

class TokenCounter:
    def __init__(self):
        self.tokenizer = None
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    def count_tokens(self, text: str) -> int:
        """Calculate text token count"""
        if not text:
            return 0
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(str(text)))
            except:
                pass
        # Fallback estimation: English ~4 chars/token, Chinese ~1.5 chars/token
        return len(str(text)) // 3

    def count_messages_tokens(self, messages: List[Dict]) -> int:
        """Calculate total tokens for a list of messages"""
        total_tokens = 0
        for msg in messages:
            content = msg.get('content', '')
            # If content is an object, convert to string for counting
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            total_tokens += self.count_tokens(str(content))
        return total_tokens

_token_counter = TokenCounter()

# ============================================================
# 2. Configuration & Data Structures
# ============================================================

@dataclass
class ModelConfig:
    name: str
    model_type: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    #BT
    model_path: Optional[str] = None      # HF repo id or local path
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_length: int = 16384               # RM context length (Skywork V2 16k)
    tie_eps: float = 1e-4
# ============================================================
# 3. Prompt Templates & Grading Rubric
# ============================================================

PAIRWISE_JUDGE_PROMPT_TEMPLATE = r"""You are an expert evaluator for AI agent systems that use tools to solve tasks.
Your task is to compare two agent trajectories (Trajectory A and Trajectory B) and determine which one better solves the user's problem.

### Input Data

**Available Tools:**
<tool_environment>
{tool_descriptions}
</tool_environment>

**User Query:**
<user_query>
{user_query}
</user_query>

**Trajectory A:**
<trajectory_a>
{trajectory_a_log}
</trajectory_a>

**Trajectory B:**
<trajectory_b>
{trajectory_b_log}
</trajectory_b>

### Evaluation Criteria
Consider the following aspects when comparing the two trajectories:
- **Correctness**: Does the agent correctly understand and fulfill the user's request?
- **Tool Usage**: Are tool calls valid, necessary, and properly parameterized?
- **Efficiency**: Does the agent avoid redundant or unnecessary steps?
- **Consistency**: Is context maintained across the conversation turns?
- **Helpfulness**: Does the final response adequately address the user's needs?

### Output Format
Output ONLY a JSON object:
{{
  "reasoning": "brief explanation of your comparison...",
  "winner": "[[A]]" or "[[B]]" or "[[Tie]]"
}}
"""

# ============================================================
# 4. Core Evaluation Class (Judge)
# ============================================================

class LLMJudgeEvaluator:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config

    def format_trajectory(self, messages: List[Dict]) -> str:
        formatted = []
        if not isinstance(messages, list):
            return str(messages)
            
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            # Try parsing JSON strings to make the Prompt more readable
            if isinstance(content, str) and content.strip().startswith('{'):
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, ensure_ascii=False, indent=2)
                except:
                    pass
            elif isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False, indent=2)
            
            formatted.append(f"[{i}] {role.upper()}: {content}")
        return '\n\n'.join(formatted)

    def format_tools(self, tools: List[Dict]) -> str:
        """
        Format tool descriptions.
        Directly dump the complete JSON Schema so the Judge can check parameter correctness (Parameter Grounding).
        """
        if not tools: return "No tools available."
        try:
            return json.dumps(tools, ensure_ascii=False, indent=2)
        except Exception:
            return str(tools)

    def build_prompt(self, query, tools, traj_a, traj_b, category):
        return PAIRWISE_JUDGE_PROMPT_TEMPLATE.format(
            tool_descriptions=self.format_tools(tools),
            user_query=query if isinstance(query, str) else str(query),
            trajectory_a_log=self.format_trajectory(traj_a),
            trajectory_b_log=self.format_trajectory(traj_b),
        )

    def _call_api(self, prompt: str) -> str:
        headers = {
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.config.name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }
        
        # Auto-complete endpoint
        endpoint = self.config.base_url.rstrip('/')
        if not endpoint.endswith('chat/completions'):
             endpoint += '/chat/completions'

        # Retry logic
        for attempt in range(3):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=120
                )
                if response.status_code == 200:
                    try:
                        return response.json()['choices'][0]['message']['content']
                    except KeyError:
                        print(f"API Response Format Error: {response.text}")
                        return ""
                elif response.status_code == 429: 
                    time.sleep(5 * (attempt + 1))
                    continue
                elif response.status_code >= 500:
                    time.sleep(2)
                    continue
                else:
                    print(f"API Error {response.status_code}: {response.text}")
                    break
            except Exception as e:
                print(f"Request failed: {e}")
                time.sleep(2)
        return ""

    def judge_pair(self, query, tools, chosen, rejected, category, swap=False):
        # If swap=True, A=Rejected, B=Chosen
        traj_a = rejected if swap else chosen
        traj_b = chosen if swap else rejected
        
        prompt = self.build_prompt(query, tools, traj_a, traj_b, category)
        response = self._call_api(prompt)
        
        # Simple parsing logic
        normalized_resp = response.upper()
        winner = "UNKNOWN"
        if '[[A]]' in normalized_resp: winner = 'A'
        elif '[[B]]' in normalized_resp: winner = 'B'
        elif '[[TIE]]' in normalized_resp: winner = 'Tie'
        # JSON fallback parsing
        elif '"WINNER": "A"' in normalized_resp: winner = 'A'
        elif '"WINNER": "B"' in normalized_resp: winner = 'B'
        elif '"WINNER": "TIE"' in normalized_resp: winner = 'Tie'
        
        return {
            "winner": winner,
            "raw_response": response,
            "swapped": swap
        }

        
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import inspect

class BTRewardEvaluator:
    """
    Pointwise Reward Model evaluator, then converted to pairwise winner by comparing two scores.

    Supports:
      - SeqCls-style reward models: AutoModelForSequenceClassification -> logits
      - InternLM2 reward-style models: AutoModel + model.get_score/get_scores (trust_remote_code)
    """

    def __init__(self, model_config):
        self.config = model_config
        assert self.config.model_path, "BT model requires model_path"

        # 1) Fixed device: Strongly recommended to use explicit form like cuda:0 / cuda:1
        self.device = torch.device(self.config.device if self.config.device else
                                   ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)  # Avoid default cuda:0 usage

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }.get(getattr(self.config, "dtype", "bfloat16"), torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            use_fast=True
        )

        # pad_token fallback
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backend = "seqcls"
        self.model = None

        # 2) Critical: Do not use device_map="auto" (causes gather device mismatch across cards)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=None,
            ).to(self.device)
            self.backend = "seqcls"
        except Exception:
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=None,
            ).to(self.device)
            self.backend = "api"

        self.model.eval()

        # 3) Critical: Disable cache to avoid from_legacy_cache error
        if hasattr(self.model, "config") and self.model.config is not None:
            self.model.config.use_cache = False
    # =========================
    # Helpers
    # =========================
    def format_tools(self, tools):
        """For runner token stats / debug: Interface aligned with LLMJudgeEvaluator."""
        try:
            return json.dumps(tools, ensure_ascii=False, indent=2)
        except Exception:
            return str(tools)

    def _convert_messages_to_chat(self, query, tools, messages):
        """
        Convert benchmark query/tools/trajectory into RM-consumable chat(list[dict]).

        Convention:
          - First user message contains tool_env + user_query (as currently done)
          - tool_call/tool_response in trajectory are encoded as assistant text lines (not real tool role)
        """
        tool_blob = self.format_tools(tools)
        header = (
            f"<tool_environment>\n{tool_blob}\n</tool_environment>\n"
            f"<user_query>\n{query}\n</user_query>"
        )

        chat = [{"role": "user", "content": header}]

        if not isinstance(messages, list):
            chat.append({"role": "assistant", "content": str(messages)})
            return chat

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            name = msg.get("name", None)

            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)

            if role == "user":
                chat.append({"role": "user", "content": str(content)})
            elif role == "assistant":
                chat.append({"role": "assistant", "content": str(content)})
            elif role in ("tool_call", "tool_response", "tool"):
                tag = role.upper()
                if name:
                    chat.append({"role": "assistant", "content": f"[{tag} name={name}] {content}"})
                else:
                    chat.append({"role": "assistant", "content": f"[{tag}] {content}"})
            else:
                chat.append({"role": "assistant", "content": f"[{role.upper()}] {content}"})

        return chat

    def _render_chat_to_text(self, chat):
        """
        Render chat to plain text (for seqcls logits or tokenizer without chat_template)
        """
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
        # fallback: Simple concatenation
        return "\n".join([f"{m['role']}: {m['content']}" for m in chat])

    def _score_via_api(self, chat) -> float:
        """
        For custom models like InternLM2-Reward:
        - Prioritize get_score(tokenizer, chat)
        - Otherwise try get_scores(tokenizer, [chat])
        - Otherwise raise error
        """
        # Common: get_score(tokenizer, chat)
        if hasattr(self.model, "get_score"):
            return float(self.model.get_score(self.tokenizer, chat))

        # Batch interface: get_scores(tokenizer, chats)
        if hasattr(self.model, "get_scores"):
            scores = self.model.get_scores(self.tokenizer, [chat])
            # scores 可能是 list/np/torch
            if isinstance(scores, (list, tuple)):
                return float(scores[0])
            if torch.is_tensor(scores):
                return float(scores.view(-1)[0].item())
            return float(scores)

        raise RuntimeError(
            "This reward model is loaded via AutoModel(trust_remote_code=True) "
            "but has no get_score/get_scores. Please check the model's README usage."
        )

    @torch.no_grad()
    def score(self, query, tools, messages) -> float:
        chat = self._convert_messages_to_chat(query, tools, messages)

        if self.backend == "api":
            # InternLM2 reward common: model.get_score / get_scores
            return self._score_via_api(chat)

        text = self._render_chat_to_text(chat)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(self.config, "max_length", 16384),
            padding=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Some SeqCls forward methods don't accept use_cache, so add compatibility
        sig = None
        try:
            sig = inspect.signature(self.model.forward)
        except Exception:
            sig = None

        if sig is not None and "use_cache" in sig.parameters:
            out = self.model(**inputs, use_cache=False)
        else:
            out = self.model(**inputs)

        return float(out.logits.view(-1)[0].item())

    def judge_pair(self, query, tools, chosen, rejected, category=None, swap=False):
        """
        Pairwise decision by comparing two pointwise scores.
        swap semantics consistent with runner: if swap=True, A=rejected, B=chosen
        """
        traj_a = rejected if swap else chosen
        traj_b = chosen if swap else rejected

        s_a = self.score(query, tools, traj_a)
        s_b = self.score(query, tools, traj_b)

        tie_eps = float(getattr(self.config, "tie_eps", 1e-4))
        if abs(s_a - s_b) <= tie_eps:
            winner = "Tie"
        else:
            winner = "A" if s_a > s_b else "B"

        return {
            "winner": winner,
            "raw_response": json.dumps({"score_a": s_a, "score_b": s_b}, ensure_ascii=False),
            "swapped": swap,
            "score_a": s_a,
            "score_b": s_b
        }
# ============================================================
# 4b. Remote BT Reward Evaluator (via serve_reward_model.py API)
# ============================================================

class RemoteBTRewardEvaluator:
    """
    Pairwise Reward Model evaluator that calls a remote serve_reward_model.py
    instance via HTTP POST /v1/score, instead of loading the model locally.

    Config requires: base_url (e.g. "http://10.22.70.84:8000"), tie_eps.
    """

    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        assert self.config.base_url, "bt_remote model requires base_url"

    def format_tools(self, tools):
        try:
            return json.dumps(tools, ensure_ascii=False, indent=2)
        except Exception:
            return str(tools)

    def _convert_messages_to_chat(self, query, tools, messages):
        """
        Convert benchmark query/tools/trajectory into a simple two-message chat:
          - user: tool_environment + user_query
          - assistant: entire trajectory flattened as text
        This guarantees strict user/assistant alternation for any chat template.
        """
        tool_blob = self.format_tools(tools)
        user_content = (
            f"<tool_environment>\n{tool_blob}\n</tool_environment>\n"
            f"<user_query>\n{query}\n</user_query>"
        )

        # Flatten the entire trajectory into a single assistant message
        trajectory_parts = []
        if not isinstance(messages, list):
            trajectory_parts.append(str(messages))
        else:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                name = msg.get("name", None)

                if isinstance(content, (dict, list)):
                    content = json.dumps(content, ensure_ascii=False)

                if role in ("tool_call", "tool_response", "tool"):
                    tag = role.upper()
                    line = f"[{tag} name={name}] {content}" if name else f"[{tag}] {content}"
                else:
                    line = f"[{role.upper()}] {content}"

                trajectory_parts.append(line)

        assistant_content = "\n\n".join(trajectory_parts)

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

    def _call_score_api(self, chat: list) -> float:
        """Call remote /v1/score endpoint to get a scalar reward score."""
        base_url = self.config.base_url or ""
        endpoint = base_url.rstrip('/') + '/v1/score'
        messages_payload = [{"role": m["role"], "content": m["content"]} for m in chat]
        data = {
            "model": self.config.name,
            "messages": messages_payload,
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    endpoint,
                    json=data,
                    timeout=300,
                )
                if response.status_code == 200:
                    result = response.json()
                    return float(result["results"][0]["score"])
                elif response.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                elif response.status_code >= 500:
                    print(f"[RemoteBT] Server error {response.status_code}: {response.text[:200]}")
                    time.sleep(3)
                    continue
                else:
                    print(f"[RemoteBT] API error {response.status_code}: {response.text[:200]}")
                    break
            except Exception as exc:
                print(f"[RemoteBT] Request failed (attempt {attempt+1}): {exc}")
                time.sleep(3)

        raise RuntimeError(f"Failed to get score from {endpoint} after 3 attempts")

    def score(self, query, tools, messages) -> float:
        chat = self._convert_messages_to_chat(query, tools, messages)
        return self._call_score_api(chat)

    def judge_pair(self, query, tools, chosen, rejected, category=None, swap=False):
        traj_a = rejected if swap else chosen
        traj_b = chosen if swap else rejected

        score_a = self.score(query, tools, traj_a)
        score_b = self.score(query, tools, traj_b)

        tie_eps = float(getattr(self.config, "tie_eps", 1e-4))
        if abs(score_a - score_b) <= tie_eps:
            winner = "Tie"
        else:
            winner = "A" if score_a > score_b else "B"

        return {
            "winner": winner,
            "raw_response": json.dumps({"score_a": score_a, "score_b": score_b}, ensure_ascii=False),
            "swapped": swap,
            "score_a": score_a,
            "score_b": score_b,
        }

# ============================================================
# 5. Evaluation Runner
# ============================================================

class BenchmarkRunner:
    # Length bin definitions
    LENGTH_BINS = [
        (0, 4000, '0-4k'),
        (4000, 8000, '4-8k'),
        (8000, 16000, '8-16k'),
        (16000, 32000, '16-32k'),
        (32000, float('inf'), '32k+')
    ]

    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.BENCHMARKS = {
            'planning_multi_easy': 'planning_multi_easy.jsonl',
            'planning_multi_hard': 'planning_multi_hard.jsonl',
            'planning_single_easy': 'planning_single_easy.jsonl',
            'planning_single_hard': 'planning_single_hard.jsonl',
            'robust_recovery': 'robust_recovery.jsonl',
            'safety_refusal': 'safety_refusal.jsonl',
            'tool_irrelevance': 'tool_irrelevance.jsonl',
        }

    def _get_messages(self, data_obj):
        """Safely extract messages list"""
        if isinstance(data_obj, dict):
            if 'messages' in data_obj:
                return data_obj['messages']
            return [] 
        if isinstance(data_obj, list):
            return data_obj
        return []

    def _judge_single_item(self, evaluator, item, category, swap):
        """Logic for processing a single item"""
        try:
            # Extract data
            chosen_msgs = self._get_messages(item['chosen'])
            reject_obj = item.get('reject', item.get('rejected'))
            reject_msgs = self._get_messages(reject_obj)

            # Call evaluation
            res = evaluator.judge_pair(
                item['query'], item.get('tools', []), 
                chosen_msgs, reject_msgs, 
                category, swap
            )
            
            # Calculate score
            score = 0.0
            if res['winner'] == 'Tie':
                score = 0.5
            elif not swap: # A is Chosen, Judge picked A
                score = 1.0 if res['winner'] == 'A' else 0.0
            else: # A is Rejected, B is Chosen, Judge picked B
                score = 1.0 if res['winner'] == 'B' else 0.0
            
            # === Token Statistics (Full Context) ===
            # 1. Trajectory
            tok_c = _token_counter.count_messages_tokens(chosen_msgs)
            tok_r = _token_counter.count_messages_tokens(reject_msgs)
            
            # 2. Tools (Count using full JSON string)
            tools_list = item.get('tools', [])
            formatted_tools = evaluator.format_tools(tools_list)
            tok_tools = _token_counter.count_tokens(formatted_tools)
            
            # 3. Query
            query_text = item.get('query', '')
            tok_query = _token_counter.count_tokens(str(query_text))
            
            total_input_tokens = tok_c + tok_r + tok_tools + tok_query
            
            return {
                "uuid": item.get('uuid'),
                "score": score,
                "winner": res['winner'],
                "swapped": swap,
                "total_tokens": total_input_tokens,
                "response": res['raw_response']
            }
        except Exception as e:
            return {"uuid": item.get('uuid'), "error": str(e), "score": 0}

    def run_benchmark(self, model_config, bench_names=None, workers=8, no_swap=False):
        if not bench_names or 'all' in bench_names: 
            bench_names = list(self.BENCHMARKS.keys())
        if model_config.model_type == "bt":
            no_swap = True
            evaluator = BTRewardEvaluator(model_config)
            # BT local inference, suggest workers=1~2 to avoid VRAM jitter/thread contention
            workers = min(workers, 2)
        elif model_config.model_type == "bt_remote":
            no_swap = True
            evaluator = RemoteBTRewardEvaluator(model_config)
            # Remote API can handle moderate concurrency
            workers = min(workers, 4)
        else:
            evaluator = LLMJudgeEvaluator(model_config)
        final_summary = {}
        
        # Model root output directory
        model_output_root = self.output_dir / model_config.name
        model_output_root.mkdir(parents=True, exist_ok=True)

        for bench_name in bench_names:
            # === 1. File location ===
            target_file = None
            if bench_name in self.BENCHMARKS:
                target_file = self.BENCHMARKS[bench_name]
            else:
                # Fuzzy matching
                for k in self.BENCHMARKS:
                    if bench_name in k:
                        target_file = self.BENCHMARKS[k]
                        bench_name = k
                        break
            
            if not target_file: continue

            file_path = self.data_dir / target_file
            # Compatible with _unified suffix
            if not file_path.exists():
                alt_path = self.data_dir / target_file.replace('.jsonl', '_unified.jsonl')
                if alt_path.exists():
                    file_path = alt_path

            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                continue

            print(f"\n🚀 Running {bench_name} ...")
            
            # === 2. Prepare output ===
            result_dir = model_output_root / bench_name
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / "predictions.jsonl"
            
            # === 3. Resume from breakpoint ===
            completed_uuids = set()
            if result_file.exists():
                with open(result_file, 'r') as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            # Key = uuid + swapped_status
                            key = f"{rec['uuid']}_{rec.get('swapped', False)}"
                            completed_uuids.add(key)
                        except: pass
            
            # === 4. Generate tasks ===
            tasks = []
            with open(file_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        item = json.loads(line)
                        # Compatible with uuid/parent_uuid
                        uuid = item.get('uuid') or item.get('parent_uuid')
                        if not uuid: continue
                        item['uuid'] = uuid 
                        
                        # Task 1: Normal Order
                        if f"{uuid}_False" not in completed_uuids:
                            tasks.append((item, False))
                        
                        # Task 2: Swapped Order
                        if not no_swap and f"{uuid}_True" not in completed_uuids:
                            tasks.append((item, True))
                    except: continue
            
            # === 5. Concurrent execution ===
            if not tasks:
                print("   All tasks completed.")
            else:
                print(f"   Tasks to run: {len(tasks)}")
                with open(result_file, 'a', encoding='utf-8') as f_out:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {
                            executor.submit(self._judge_single_item, evaluator, item, bench_name, swap): item['uuid']
                            for item, swap in tasks
                        }
                        
                        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                            res = future.result()
                            if "error" not in res:
                                f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                                f_out.flush()
                            else:
                                print(f"Error processing {res.get('uuid')}: {res.get('error')}")
            
            # === 6. Statistics and Saving ===
            stats = self._analyze_benchmark(result_file, bench_name)
            final_summary[bench_name] = stats
            
            # Save individual metrics.json
            metrics_file = result_dir / "metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            # Refresh global all_results.json
            agg_file = model_output_root / "all_results.json"
            with open(agg_file, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, ensure_ascii=False, indent=2)
            
        self._print_summary(final_summary)

    def _analyze_benchmark(self, result_file, bench_name):
        """Analyze result file, calculate accuracy and bin statistics"""
        scores = []
        ties = 0
        bins = defaultdict(list)
        total_count = 0
        
        with open(result_file, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "error" in d: continue
                    total_count += 1
                    
                    s = d['score']
                    scores.append(s)
                    if d.get('winner') == 'Tie': ties += 1
                    
                    t = d.get('total_tokens', 0)
                    for low, high, name in self.LENGTH_BINS:
                        if low <= t < high:
                            bins[name].append(s)
                            break
                except: pass
        
        # Use float() conversion to ensure JSON serializability
        avg_acc = float(np.mean(scores)) if scores else 0.0
        tie_rate = (ties / len(scores)) if scores else 0.0
        
        bin_stats = {}
        for k, v in bins.items():
            bin_stats[k] = {
                "accuracy": float(np.mean(v)),
                "count": len(v)
            }
        
        return {
            "accuracy": avg_acc * 100,
            "tie_rate": tie_rate * 100,
            "total_samples": total_count,
            "length_breakdown": bin_stats
        }

    def _print_summary(self, summary):
        print("\n" + "="*65)
        print(f"{'Benchmark':<25} | {'Acc %':<10} | {'Tie %':<10} | {'Count':<6}")
        print("-" * 65)
        
        total_acc = []
        for name, stats in summary.items():
            print(f"{name:<25} | {stats['accuracy']:6.2f}     | {stats['tie_rate']:6.2f}     | {stats['total_samples']:<6}")
            total_acc.append(stats['accuracy'])
            
        if total_acc:
            print("-" * 65)
            print(f"{'AVERAGE':<25} | {np.mean(total_acc):6.2f}     | -          | -")
        print("="*65)

# ============================================================
# 6. Main Entry (Modified to add --models support)
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to models.yaml')
    parser.add_argument('--data-dir', type=str, required=True, help='Root dir containing jsonl files')
    parser.add_argument('--output-dir', type=str, default='./eval_results')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--benchmarks', nargs='+', default=None)
    parser.add_argument('--no-swap', action='store_true', help='Disable position swap (run only A vs B)')
    
    parser.add_argument('--models', nargs='+', help='Specify model names to run (must match names in config.yaml)')
    
    args = parser.parse_args()

    # Read Config
    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    
    runner = BenchmarkRunner(args.data_dir, args.output_dir)

    #  Model filtering logic
    models_to_run = conf['models']
    if args.models:
        print(f"Filtering models: {args.models}")
        models_to_run = [m for m in conf['models'] if m['name'] in args.models]
        if not models_to_run:
            print(f"❌ Error: No matching models found in config.yaml for: {args.models}")
            print("Available models:", [m['name'] for m in conf['models']])
            return

    for m in models_to_run:
        # Environment variable replacement
        api_key = m.get('api_key', '')
        if api_key and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.getenv(env_var)
            if not api_key:
                print(f"Warning: Env var {env_var} not set for model {m['name']}")
        
        model_cfg = ModelConfig(
            name=m['name'],
            model_type=m['model_type'],
            api_key=api_key,
            base_url=m.get('base_url'),
            temperature=m.get('temperature', 0.0),
            max_tokens=m.get('max_tokens', 4096),
            model_path=m.get('model_path'),
            device=m.get('device', 'cuda'),
            dtype=m.get('dtype', 'bfloat16'),
            max_length=m.get('max_length', 16384),
            tie_eps=m.get('tie_eps', 1e-4)
        )
        
        runner.run_benchmark(model_cfg, args.benchmarks, args.workers, args.no_swap)

if __name__ == "__main__":
    main()