
"""
 Evaluation script using IG policy.
Supports GPT (OpenAI), HuggingFace, and Mistral models.
Runs on test queries and saves results to CSV and JSON.
Routing is enabled with top_k=5.
"""

from __future__ import annotations

import json
import csv
import argparse
import warnings
import sys
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from contextlib import contextmanager
from io import StringIO
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass

from openai import OpenAI
import requests

# HuggingFace imports (optional, only needed for HF models)
try:
    import torch
    import transformers
    from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
    from transformers.utils import logging as hf_logging
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import duckdb

# Import local modules
import uq
import route_solver
import llm_dialog_sim
import ig_policy
import prompts
import dataset_loader
import paths

# Use paths from paths.py
BASE = paths.BASE
RESULT_DIR = paths.RESULT_DIR
FINAL_DATA = paths.FINAL_DATA
FULL_DATA = paths.FULL_DATA
TEST_QUERIES_FILE = paths.TEST_QUERIES_FILE
EVAL_QUERY_CSV = paths.EVAL_QUERY_CSV
DIALOGUE_FILE = paths.DIALOGUE_FILE
TRUE_INTENT_FILE = paths.TRUE_INTENT_FILE
SCRIPTS_DIR = paths.SCRIPTS_DIR
ENV_FILE = paths.ENV_FILE

# Routing configuration
TOP_K = 5  # Hardcoded top_k for routing

# Min prob values file path
MIN_PROB_VALUES_FILE = paths.MIN_PROB_VALUES_FILE

warnings.filterwarnings("ignore", category=FutureWarning)


def load_min_prob_values() -> Tuple[float, float]:
    """
    Load min_prob values from min_prob_values.txt file.
    
    Returns:
        (target_min_prob, mode_min_prob)
    """
    default_target = 0.2
    default_mode = 0.5
    
    if not MIN_PROB_VALUES_FILE.exists():
        print(f"Warning: {MIN_PROB_VALUES_FILE} not found, using defaults: target={default_target}, mode={default_mode}")
        return default_target, default_mode
    
    target_min_prob = default_target
    mode_min_prob = default_mode
    
    try:
        with open(MIN_PROB_VALUES_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'target_min_prob':
                        target_min_prob = float(value)
                    elif key == 'mode_min_prob':
                        mode_min_prob = float(value)
        
        # Clamp values to [0, 1]
        target_min_prob = max(0.0, min(1.0, target_min_prob))
        mode_min_prob = max(0.0, min(1.0, mode_min_prob))
        
    except Exception as e:
        print(f"Warning: Failed to load min_prob values from {MIN_PROB_VALUES_FILE}: {e}")
        print(f"Using defaults: target={default_target}, mode={default_mode}")
    
    return target_min_prob, mode_min_prob


@contextmanager
def suppress_debug_output():
    """Context manager to suppress debug print statements."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def format_poi_result(poi: Dict[str, Any]) -> Dict[str, Any]:
    """Format POI result in the requested format."""
    return {
        "poi_name": poi.get("name") or poi.get("poi_id", "NONE"),
        "poi_type": poi.get("poi_type") or poi.get("fclass", "unknown"),
        "route_cost_s": poi.get("route_cost_s") or poi.get("travel_time_s") or 0.0,
        "distance_m": poi.get("distance_m") or 0.0,
    }


# ============================================================================
# Model-specific client initialization
# ============================================================================

def get_openai_client(model_name: str) -> OpenAI:
    """Get OpenAI client for GPT models."""
    load_dotenv(str(ENV_FILE))
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY env var.")
    
    # Handle Qwen models that use OpenAI-compatible API
    if model_name in ["qwen3", "qwen3-32b"]:
        base_url = os.environ.get("OPENAI_API_BASE")
        if not base_url:
            raise SystemExit("OPENAI_API_BASE must be set for qwen3 model")
        return OpenAI(base_url=base_url, api_key=api_key)
    
    base_url = os.environ.get("OPENAI_API_BASE")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


class MistralClient:
    """Wrapper class that mimics OpenAI client interface but uses Mistral API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chat = self.Chat(api_key)
    
    class Completions:
        """Completions class that provides the create() method."""
        def __init__(self, api_key: str):
            self.api_key = api_key
        
        def create(self, model: str, messages: list, temperature: float = 0.0, max_tokens: int = 500, **kwargs):
            """Create a chat completion using Mistral API."""
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "ministral-8b-2512",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            
            if resp.status_code != 200:
                print(f"\n[Mistral API Error] Status Code: {resp.status_code}", flush=True)
                print(f"[Mistral API Error] URL: {resp.url}", flush=True)
                try:
                    error_body = resp.text
                    print(f"[Mistral API Error] Response Body: {error_body}", flush=True)
                except Exception:
                    print(f"[Mistral API Error] Could not read response body", flush=True)
            
            resp.raise_for_status()
            response_data = resp.json()
            
            # Return the actual response data - will be handled by model-specific code
            return response_data
    
    class Chat:
        """Chat class that provides the completions attribute."""
        def __init__(self, api_key: str):
            self.completions = MistralClient.Completions(api_key)


def get_mistral_client() -> MistralClient:
    """Get Mistral API client wrapper."""
    load_dotenv(str(ENV_FILE))
    api_key = os.environ.get("Mistral_key")
    if not api_key:
        raise SystemExit("Missing Mistral_key env var.")
    return MistralClient(api_key)


# HuggingFace model support
if HF_AVAILABLE:
    class HFRunner:
        """Small wrapper around (tokenizer, model) to do GenerationConfig-based generation."""
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def generate_text(self, prompt: str, gen_config: GenerationConfig) -> str:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
            out = self.model.generate(**inputs, generation_config=gen_config)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]
            return text.strip()

    def _gen_config(pipeline, *, do_sample: bool, max_new_tokens: int, temperature: Optional[float] = None) -> GenerationConfig:
        """Create GenerationConfig with pad/eos ids to avoid HF warnings."""
        tok = getattr(pipeline, "tokenizer", None)
        pad_id = getattr(tok, "pad_token_id", None)
        eos_id = getattr(tok, "eos_token_id", None)
        cfg = GenerationConfig(
            do_sample=bool(do_sample),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        if temperature is not None and do_sample:
            cfg.temperature = float(temperature)
        return cfg

    def extract_json_object(text: str) -> Dict[str, Any]:
        """Extract JSON object from LLM response."""
        s = (text or "").strip()
        if not s:
            raise ValueError("Empty response from LLM")

        if s.startswith("```"):
            lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
            s = "\n".join(lines).strip()

        first = s.find("{")
        if first != -1:
            brace_count = 0
            end_pos = -1
            for i, ch in enumerate(s[first:], start=first):
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            if end_pos > first:
                outer = s[first:end_pos]
                try:
                    return json.loads(outer)
                except json.JSONDecodeError:
                    pass

        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not find valid JSON object in LLM response")

    def get_hf_pipeline(model_name: str, hf_token: Optional[str] = None) -> HFRunner:
        """Load HF model+tokenizer and return an HFRunner."""
        model_map = {
            "qwen3": "Qwen/Qwen3-32B",
            "qwen3-14b": "Qwen/Qwen3-14B",
            "qwen-2.5-14b": "Qwen/Qwen2.5-14B",
            "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        }
        model_name_lower = str(model_name).lower()
        model_id = model_map.get(model_name_lower, model_name)
        
        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if hf_token:
            model_kwargs["token"] = hf_token

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=model_kwargs.get("token"),
                padding_side="left",
                trust_remote_code=True,
            )
        except ValueError as e:
            msg = str(e)
            if "Qwen2Tokenizer" in msg or "tokenizer class" in msg.lower():
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    model_id,
                    token=model_kwargs.get("token"),
                    padding_side="left",
                    trust_remote_code=True,
                )
            else:
                raise
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                **{k: v for k, v in model_kwargs.items() if k != "token"},
                token=model_kwargs.get("token"),
            ).eval()
        except KeyError as e:
            if str(e).strip("'\"") == "qwen2":
                raise SystemExit(
                    "Your installed 'transformers' does not support Qwen2/Qwen2.5 models.\n"
                    "Fix: upgrade transformers or use --model qwen3 / llama3.1-8b.\n"
                    f"Model requested: {model_id}"
                )
            raise

        return HFRunner(model=model, tokenizer=tokenizer)

    class _ChoiceMsg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _ChoiceMsg(content)

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    class HFPipelineClient:
        """Minimal OpenAI-like client wrapper so we can reuse uq.build_uq_debug_info() unchanged."""
        def __init__(self, pipeline):
            self.pipeline = pipeline
            self.chat = self
            self.completions = self

        def create(self, model: str = None, messages: list = None, temperature: float = 0.0, max_tokens: int = None, **kwargs):
            prompt = ""
            for msg in (messages or []):
                if msg.get("role") == "user":
                    prompt += (("\n\n" if prompt else "") + (msg.get("content") or ""))

            if temperature and float(temperature) > 0:
                gen_config = _gen_config(
                    self.pipeline,
                    do_sample=True,
                    temperature=float(temperature),
                    max_new_tokens=int(max_tokens or 50),
                )
            else:
                gen_config = _gen_config(
                    self.pipeline,
                    do_sample=False,
                    max_new_tokens=int(max_tokens or 256),
                )

            txt = self.pipeline.generate_text(prompt, gen_config)
            return _Resp(txt)


def get_client_for_model(model_name: str):
    """Get appropriate client based on model name."""
    model_lower = model_name.lower()
    
    # HuggingFace models
    hf_models = ["llama", "qwen"]
    if any(hf in model_lower for hf in hf_models) and HF_AVAILABLE:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        pipeline = get_hf_pipeline(model_name, hf_token)
        return HFPipelineClient(pipeline)
    
    # Mistral models
    if "mistral" in model_lower or "ministral" in model_lower:
        return get_mistral_client()
    
    # Default: OpenAI/GPT models
    return get_openai_client(model_name)


# ============================================================================
# Model-specific query parsing
# ============================================================================

def parse_query(client, query: str, model: str, model_type: str) -> Tuple[Dict[str, Any], str]:
    """Parse query using INFO_PARSING_PROMPT. Works with all model types."""
    prompt = prompts.INFO_PARSING_PROMPT.format(query=query)
    
    if model_type == "hf":
        # HuggingFace model
        base_prompt = prompt
        model_name = str(client.pipeline.model.config.name_or_path).lower()
        is_instruct = ("instruct" in model_name) or ("chat" in model_name)
        
        if not is_instruct:
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Output ONLY the JSON object. Do not include any explanations, comments, or text before or after the JSON. Start with { and end with }."
            )
        
        gen_config = _gen_config(client.pipeline, do_sample=False, max_new_tokens=512)
        txt = client.pipeline.generate_text(prompt, gen_config)
        
        print(f"\n[INFO_PARSING_PROMPT Raw Response]")
        print(f"{'='*80}")
        print(txt)
        print(f"{'='*80}\n")
        
        try:
            parsed = extract_json_object(txt)
            return parsed, txt
        except ValueError as e:
            print(f"  ⚠️  JSON extraction failed: {e}")
            return {}, txt
    
    elif model_type == "mistral":
        # Mistral models - handle actual API response format
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                # Mistral returns dict directly, extract content from response structure
                if isinstance(resp, dict):
                    raw = (resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                else:
                    # Fallback for OpenAI-compatible response
                    raw = (resp.choices[0].message.content or "").strip()
                
                # Extract JSON from response
                import re
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    obj = json.loads(raw[start:end+1])
                    return obj, raw
                raise ValueError("No JSON found in response")
            except Exception as e:
                last_err = e
                if attempt < 5:
                    import time
                    time.sleep(min(2.0**attempt, 10.0))
                else:
                    break
    else:
        # OpenAI/GPT models
        last_err: Exception | None = None
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw = (resp.choices[0].message.content or "").strip()
                # Extract JSON from response
                import re
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    obj = json.loads(raw[start:end+1])
                    return obj, raw
                raise ValueError("No JSON found in response")
            except Exception as e:
                last_err = e
                if attempt < 5:
                    import time
                    time.sleep(min(2.0**attempt, 10.0))
                else:
                    break
        
        # Fallback: return empty structure
        return {
            "anchor_location": "UNKNOWN",
            "target_poi_type": {},
            "travel_threshold_meters": None,
            "navigation_mode": {},
        }, ""


def get_model_type(model_name: str) -> str:
    """Determine model type: 'gpt', 'hf', or 'mistral'."""
    model_lower = model_name.lower()
    
    if any(hf in model_lower for hf in ["llama", "qwen"]):
        return "hf"
    if "mistral" in model_lower or "ministral" in model_lower:
        return "mistral"
    return "gpt"


# ============================================================================
# Main query processing function
# ============================================================================

def run_one_query(
    client,
    query_id: str,
    query: str,
    record: Dict[str, Any],
    dialogue_record: Optional[Dict[str, Any]],
    llm_model: str,
    model_type: str,
    max_turns: int,
    scenario: str,
    target_min_prob: float,
    mode_min_prob: float,
    city: Optional[str] = None,
    skip_routing: bool = False,
) -> Dict[str, Any]:
    """Run IG policy on one query and return results."""
    
    # Detect city from record if not provided
    if city is None:
        city = record.get("city", "Chicago")
    
    # Map city name to city abbreviation
    city_abbr_map = {
        "Chicago": "CH",
        "New York": "NY",
        "San Antonio": "SA",
        "New Orleans": "NO",
        "Miami": "MI",
        "Houston": "HO",
    }
    city_abbr = city_abbr_map.get(city, city.replace(" ", "_")[:2].upper())
    
    result = {
        "query_id": query_id,
        "query": query,
        "initial_parsed_intent": None,
        "identified_entropies": [],
        "pred_ambiguity": False,
        "anchor_cluster_count": 0,
        "threshold_cluster_count": 0,
        "cluster_list": [],
        "dialogue": [],
        "num_turns": 0,
        "answers": [],
        "final_intent": None,
        "generated_sql": None,
        "sql_execution_ok": False,
        "num_candidates": 0,
        "candidate_list": [],
    }
    
    try:
        # Print query_id, query, true ambiguity
        if dialogue_record:
            gt_amb_types = dialogue_record.get("ambiguity_types", [])
            gt_is_ambiguous = len(gt_amb_types) > 0 and "Unambiguous" not in gt_amb_types
            print(f"\nTrue Ambiguity: {gt_amb_types if gt_is_ambiguous else ['Unambiguous']}")
        else:
            print(f"\nTrue Ambiguity: [Not available]")
        
        # 1. Initial Parse
        parsed, raw_response = parse_query(client, query, llm_model, model_type)
        result["initial_parsed_intent"] = parsed
        
        print(f"\n{'='*80}")
        print("1. INITIAL PARSE")
        print(f"{'='*80}")
        print(f"\nRaw Parse LLM Response:")
        print(raw_response)
        print(f"\nParsed Intent:")
        print(json.dumps(parsed, indent=2))
        
        # 2. Build UQ Debug Info (includes threshold sampling)
        print(f"\n{'='*80}")
        print("2. UNCERTAINTY QUANTIFICATION & THRESHOLD SAMPLING")
        print(f"{'='*80}")
        
        debug_info = uq.build_uq_debug_info(
            client,
            parsed,
            city_abbr=city_abbr,
            target_min_prob=target_min_prob,
            mode_min_prob=mode_min_prob,
            threshold_samples_n=10,
            threshold_eps_m=2000.0,
            threshold_temperature=0.9,
            anchor_eps_km=2.0,
            query=query,
        )
        
        anchor_clusters = debug_info.get("anchor_clusters", []) or []
        threshold_clusters = debug_info.get("threshold_clusters", []) or []
        
        # Print CP thresholds being used (from .txt file)
        cp_min_target = debug_info.get("cp_min_prob_target", target_min_prob)
        cp_min_mode = debug_info.get("cp_min_prob_mode", mode_min_prob)
        print(f"\nCP Thresholds (Conformal Prediction - from {MIN_PROB_VALUES_FILE.name}):")
        print(f"  Target min_prob: {cp_min_target:.3f}")
        print(f"  Mode min_prob: {cp_min_mode:.3f}")
        
        # Show number of anchor candidates from DB
        anchor_text = str(parsed.get("anchor_location") or "").strip()
        try:
            if anchor_text and anchor_text.lower() != "unknown":
                rows = uq.anchor_candidates_from_db(city_abbr, anchor_text)
                print(f"\nAnchor candidates from DB for '{anchor_text}': {len(rows)}")
            else:
                print("\nAnchor candidates from DB: 0 (no anchor_location parsed)")
        except Exception as e:
            print(f"\n⚠️  Failed to fetch anchor candidates from DB: {e}")
        
        # Print anchor clusters
        print("\nAnchor clusters:")
        if not anchor_clusters:
            print("  (none)")
        else:
            for c in anchor_clusters:
                name = c.get("name", "N/A")
                lat = c.get("lat", "N/A")
                lon = c.get("lon", "N/A")
                mass = c.get("mass", 0.0)
                cluster_id = c.get("id", "N/A")
                print(f"  - {cluster_id}: {name} ({lat}, {lon}) - mass={mass:.3f}")
        
        # Print threshold clusters
        print("\nThreshold clusters:")
        if not threshold_clusters:
            print("  (none)")
        else:
            for c in threshold_clusters:
                print(f"  - {c.get('id')}: {c.get('threshold_m')} m (mass={c.get('mass')})")
        
        result["anchor_cluster_count"] = len(anchor_clusters)
        result["threshold_cluster_count"] = len(threshold_clusters)
        result["cluster_list"] = {
            "anchor_clusters": [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "lat": c.get("lat"),
                    "lon": c.get("lon"),
                    "mass": c.get("mass"),
                }
                for c in anchor_clusters
            ],
            "threshold_clusters": [
                {
                    "id": c.get("id"),
                    "threshold_m": c.get("threshold_m"),
                    "mass": c.get("mass"),
                }
                for c in threshold_clusters
            ],
        }
        
        # 3. Build Marginals and Compute Entropy
        print(f"\n{'='*80}")
        print("3. ENTROPY & INFORMATION GAIN")
        print(f"{'='*80}")
        
        anchor_probs, target_probs, threshold_probs, mode_probs = ig_policy.build_marginals_from_belief_debug_info(
            debug_info, parsed
        )
        
        entropy_before = {
            "anchor": ig_policy.compute_entropy(anchor_probs) if anchor_probs else 0.0,
            "target": ig_policy.compute_entropy(target_probs) if target_probs else 0.0,
            "threshold": ig_policy.compute_entropy(threshold_probs) if threshold_probs else 0.0,
            "mode": ig_policy.compute_entropy(mode_probs) if mode_probs else 0.0,
        }
        
        print(f"\nEntropy (before clarification):")
        print(f"  Anchor: {entropy_before['anchor']:.3f} bits")
        print(f"  Target: {entropy_before['target']:.3f} bits")
        print(f"  Threshold: {entropy_before['threshold']:.3f} bits")
        print(f"  Mode: {entropy_before['mode']:.3f} bits")
        print(f"  Total: {sum(entropy_before.values()):.3f} bits")
        
        # Identify ambiguous slots: ambiguous if cluster count > 1 OR entropy >= 1.0
        identified = []
        if len(anchor_clusters) > 1 or entropy_before["anchor"] >= 1.0:
            identified.append("Anchor")
        if len(target_probs) > 1 or entropy_before["target"] >= 1.0:
            identified.append("Target")
        if len(threshold_clusters) > 1 or entropy_before["threshold"] >= 1.0:
            identified.append("Threshold")
        if mode_probs and (len(mode_probs) > 1 or entropy_before["mode"] >= 1.0):
            identified.append("Mode")
        
        result["identified_slots"] = identified
        result["identified_entropies"] = [
            {"slot": slot, "entropy": entropy_before[slot.lower()]}
            for slot in identified
        ]
        result["pred_ambiguity"] = len(identified) > 0
        
        print(f"\nIdentified ambiguity slots: {identified}")
        
        true_intent = record.get("true_intent") or {}
        
        # 4. Dialogue Simulation
        print(f"\n{'='*80}")
        print("4. DIALOGUE SIMULATION")
        print(f"{'='*80}")
        
        conversation = []
        current_anchor_probs = anchor_probs.copy()
        current_target_probs = target_probs.copy()
        current_threshold_probs = threshold_probs.copy()
        current_mode_probs = mode_probs.copy() if mode_probs else {}
        
        # IG policy configuration
        lambda_anchor = 0.0
        lambda_target = 0.0
        lambda_threshold = 0.0
        lambda_mode = 0.0
        min_score = -0.1
        tau_bits = 0.1
        p_stop = 0.95
        
        for turn_num in range(1, max_turns + 1):
            current_entropy = {
                "anchor": ig_policy.compute_entropy(current_anchor_probs) if current_anchor_probs else 0.0,
                "target": ig_policy.compute_entropy(current_target_probs) if current_target_probs else 0.0,
                "threshold": ig_policy.compute_entropy(current_threshold_probs) if current_threshold_probs else 0.0,
                "mode": ig_policy.compute_entropy(current_mode_probs) if current_mode_probs else 0.0,
            }
            
            if all(e < 0.01 for e in current_entropy.values()):
                break
            
            question_result = ig_policy.select_question(
                current_anchor_probs,
                current_target_probs,
                current_threshold_probs,
                current_mode_probs,
            )

            if not question_result or question_result.get("action") != "AskQuestion":
                break
            
            question_type = question_result.get("question_type")
            slot = {
                "AskAnchor": "anchor",
                "AskTarget": "target",
                "AskThreshold": "threshold",
                "AskMode": "mode",
            }.get(question_type, "unknown")
            
            # Generate question based on model type
            if model_type == "hf":
                # For HF models, use a simple question format
                question_text = f"Which {slot}?"
            else:
                question_text = llm_dialog_sim.assistant_question_llm(client, slot, [], model=llm_model)
                if isinstance(question_text, dict):
                    question_text = question_text.get("question", f"Which {slot}?")
            
            options = []
            if question_type == "AskAnchor":
                options = [f"{c['id']}" for c in anchor_clusters]
            elif question_type == "AskTarget":
                options = list(current_target_probs.keys())
            elif question_type == "AskThreshold":
                options = [f"{c['id']}" for c in threshold_clusters]
            elif question_type == "AskMode":
                options = list(current_mode_probs.keys())
            
            # Get oracle answer
            ti_for_oracle = {}
            if true_intent:
                ti_for_oracle = {
                    "target_label": true_intent.get("target_type"),
                    "mode_label": true_intent.get("nav_mode"),
                    "anchor_lat": true_intent.get("anchor", {}).get("lat"),
                    "anchor_lon": true_intent.get("anchor", {}).get("lon"),
                    "threshold_m": true_intent.get("distance_threshold_km"),
                }
                if ti_for_oracle["threshold_m"] is not None:
                    ti_for_oracle["threshold_m"] = float(ti_for_oracle["threshold_m"]) * 1000.0
            
            cluster_id = llm_dialog_sim.oracle_answer(
                question_type,
                options,
                ti_for_oracle,
                anchor_clusters,
                threshold_clusters,
            )
            
            # Use true intent value as answer (not cluster ID) for threshold and anchor
            answer = cluster_id  # Default to cluster ID
            if true_intent:
                if question_type == "AskTarget":
                    true_target = true_intent.get("target_type")
                    if true_target:
                        answer = true_target
                elif question_type == "AskMode":
                    true_mode = true_intent.get("nav_mode")
                    if true_mode:
                        answer = true_mode if true_mode in options else cluster_id
                elif question_type == "AskThreshold":
                    true_threshold_km = true_intent.get("distance_threshold_km")
                    if true_threshold_km is not None:
                        true_threshold_m = float(true_threshold_km) * 1000.0
                        answer = str(true_threshold_m)
                elif question_type == "AskAnchor":
                    true_anchor = true_intent.get("anchor", {})
                    anchor_lat = true_anchor.get("lat")
                    anchor_lon = true_anchor.get("lon")
                    if anchor_lat is not None and anchor_lon is not None:
                        answer = f"{anchor_lat},{anchor_lon}"
            
            # Update probabilities using cluster_id (for matching with probability distributions)
            prob_update_key = cluster_id if question_type in ["AskAnchor", "AskThreshold"] else answer
            if question_type == "AskAnchor" and prob_update_key:
                current_anchor_probs = {prob_update_key: 1.0}
            elif question_type == "AskTarget" and answer:
                current_target_probs = {answer: 1.0}
            elif question_type == "AskThreshold" and prob_update_key:
                current_threshold_probs = {prob_update_key: 1.0}
            elif question_type == "AskMode" and answer:
                current_mode_probs = {answer: 1.0}
            
            conversation.append({
                "turn": turn_num,
                "question_type": question_type,
                "question_text": question_text,
                "answer": answer,
            })
            result["answers"].append(answer)
        
        result["dialogue"] = conversation
        result["num_turns"] = len(conversation)
        
        print(f"\nDialogue turns: {len(conversation)}")
        if conversation:
            for turn_data in conversation:
                turn_num = turn_data.get("turn", 0)
                question_type = turn_data.get("question_type", "Unknown")
                question_text = turn_data.get("question_text", f"Which {question_type.lower().replace('ask', '')}?")
                answer = turn_data.get("answer", "N/A")
                print(f"  Turn {turn_num}:")
                print(f"    Q ({question_type}): {question_text}")
                print(f"    A: {answer}")
        else:
            print("  (No dialogue turns)")
        
        # 5. Final Intent
        print(f"\n{'='*80}")
        print("5. FINAL INTENT")
        print(f"{'='*80}")
        
        def _argmax_with_random_tie(probs):
            if not probs:
                return None
            max_val = max(probs.values())
            candidates = [k for k, v in probs.items() if v == max_val]
            return random.choice(candidates) if len(candidates) > 1 else candidates[0]
        
        asked_threshold = any(t["question_type"] == "AskThreshold" for t in conversation)
        asked_anchor = any(t["question_type"] == "AskAnchor" for t in conversation)
        asked_target = any(t["question_type"] == "AskTarget" for t in conversation)
        asked_mode = any(t["question_type"] == "AskMode" for t in conversation)
        
        # Get initial parsed values for argmax when no question was asked
        initial_target_probs = parsed.get("target_poi_type", {})
        if isinstance(initial_target_probs, dict):
            initial_target_probs_normalized = uq.complete_and_normalize(initial_target_probs, uq.TARGET_TYPES)
        else:
            initial_target_probs_normalized = {}
        
        initial_mode_probs = parsed.get("navigation_mode", {})
        if isinstance(initial_mode_probs, dict):
            initial_mode_probs_normalized = uq.complete_and_normalize(initial_mode_probs, uq.NAV_MODES)
        else:
            initial_mode_probs_normalized = {}
        
        # Target: use true value if asked, otherwise argmax from initial parsed intent
        if asked_target and true_intent:
            best_target = true_intent.get("target_type")
        elif initial_target_probs_normalized:
            best_target = _argmax_with_random_tie(initial_target_probs_normalized)
        else:
            best_target = _argmax_with_random_tie(current_target_probs) if current_target_probs else None
        
        # Mode: use true value if asked, otherwise argmax from initial parsed intent
        if asked_mode and true_intent:
            best_mode = true_intent.get("nav_mode", "drive")
        elif initial_mode_probs_normalized:
            best_mode = _argmax_with_random_tie(initial_mode_probs_normalized) or "drive"
        else:
            best_mode = _argmax_with_random_tie(current_mode_probs) if current_mode_probs else "drive"
        
        # Anchor and threshold: use current_probs (updated during dialogue if asked)
        best_anchor_id = _argmax_with_random_tie(current_anchor_probs) if current_anchor_probs else None
        best_threshold_id = _argmax_with_random_tie(current_threshold_probs) if current_threshold_probs else None
        
        best_anchor_cluster = next((a for a in anchor_clusters if a["id"] == best_anchor_id), None)
        best_threshold_cluster = next((t for t in threshold_clusters if t["id"] == best_threshold_id), None)
        
        # Threshold: use true value if asked
        parsed_threshold_m = parsed.get("travel_threshold_meters")
        if asked_threshold and true_intent:
            true_threshold_km = true_intent.get("distance_threshold_km")
            if true_threshold_km is not None:
                true_threshold_m = float(true_threshold_km) * 1000.0
                final_threshold = {
                    "id": best_threshold_id,
                    "threshold_m": true_threshold_m,
                }
            else:
                final_threshold = {
                    "id": best_threshold_id,
                    "threshold_m": best_threshold_cluster.get("threshold_m") if best_threshold_cluster else None,
                }
        elif not asked_threshold and parsed_threshold_m is not None:
            final_threshold = {
                "id": None,
                "threshold_m": float(parsed_threshold_m),
            }
        else:
            final_threshold = {
                "id": best_threshold_id,
                "threshold_m": best_threshold_cluster.get("threshold_m") if best_threshold_cluster else None,
            }
        
        # Anchor: use true coordinates if asked
        if asked_anchor and true_intent:
            true_anchor = true_intent.get("anchor", {})
            anchor_lat = true_anchor.get("lat")
            anchor_lon = true_anchor.get("lon")
            anchor_name = true_anchor.get("name")
            if anchor_lat is not None and anchor_lon is not None:
                final_anchor = {
                    "id": best_anchor_id,
                    "name": anchor_name if anchor_name else (best_anchor_cluster.get("name") if best_anchor_cluster else None),
                    "lat": float(anchor_lat),
                    "lon": float(anchor_lon),
                }
            else:
                final_anchor = {
                    "id": best_anchor_id,
                    "name": best_anchor_cluster.get("name") if best_anchor_cluster else None,
                    "lat": best_anchor_cluster.get("lat") if best_anchor_cluster else None,
                    "lon": best_anchor_cluster.get("lon") if best_anchor_cluster else None,
                }
        else:
            final_anchor = {
                "id": best_anchor_id,
                "name": best_anchor_cluster.get("name") if best_anchor_cluster else None,
                "lat": best_anchor_cluster.get("lat") if best_anchor_cluster else None,
                "lon": best_anchor_cluster.get("lon") if best_anchor_cluster else None,
            }
        
        final_intent = {
            "anchor": final_anchor,
            "target": best_target,
            "threshold": final_threshold,
            "mode": best_mode,
        }
        result["final_intent"] = final_intent
        print(f"\nFinal Intent:")
        print(json.dumps(final_intent, indent=2))
        
        # 6. Generate SQL
        print(f"\n{'='*80}")
        print("6. SQL GENERATION")
        print(f"{'='*80}")
        
        anchor_lat = final_intent.get("anchor", {}).get("lat")
        anchor_lon = final_intent.get("anchor", {}).get("lon")
        final_target = final_intent.get("target")
        threshold_m = final_intent.get("threshold", {}).get("threshold_m")
        
        # Check if anchor location is null in parsed intent or final intent
        parsed_anchor_location = parsed.get("anchor_location")
        anchor_location_is_null = (
            parsed_anchor_location is None or 
            str(parsed_anchor_location).strip().lower() in ["unknown", "", "null", "none"]
        )
        final_anchor_is_null = anchor_lat is None or anchor_lon is None
        
        if anchor_location_is_null or final_anchor_is_null:
            print(f"\nSkipping SQL generation: anchor location is null")
            print(f"  - Parsed anchor_location: {parsed_anchor_location}")
            print(f"  - Final anchor lat: {anchor_lat}, lon: {anchor_lon}")
            result["generated_sql"] = None
            result["sql_execution_ok"] = False
            result["num_candidates"] = 0
            result["candidate_list"] = []
            # Do not set sql_error - it's not a SQL mistake
        else:
            print(f"\nCalling llm_text_to_sql with final_intent:")
            print(f"  - anchor_lat: {anchor_lat}")
            print(f"  - anchor_lon: {anchor_lon}")
            print(f"  - final_target: {final_target}")
            print(f"  - threshold_m: {threshold_m}")
            
            try:
                sql = route_solver.llm_text_to_sql(
                    client,
                    city_abbr=city_abbr,
                    anchor_lat=anchor_lat,
                    anchor_lon=anchor_lon,
                    target_label=final_target if final_target else None,
                    threshold_m=threshold_m,
                    model=llm_model,
                )
                result["generated_sql"] = sql
                print(f"\nGenerated SQL:")
                print(sql)
            except Exception as e:
                result["generated_sql"] = f"SQL generation failed: {str(e)}"
                result["sql_execution_ok"] = False
                result["sql_error"] = str(e)
                print(f"\nSQL Generation Failed: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Execute SQL and get candidates (only if SQL was generated successfully)
            if result.get("generated_sql") and not result["generated_sql"].startswith("SQL generation failed"):
                candidates = []
                try:
                    con = duckdb.connect(str(route_solver.DB_PATH), read_only=True)
                    con.execute("INSTALL spatial;")
                    con.execute("LOAD spatial;")
                    candidates = con.execute(result["generated_sql"]).fetchall()
                    con.close()
                    
                    # Convert to list of dicts
                    candidate_list = []
                    if candidates:
                        candidate_list = [
                            {
                                "name": row[0] if len(row) > 0 else "N/A",
                                "fclass": row[1] if len(row) > 1 else "N/A",
                                "lat": float(row[2]) if len(row) > 2 and row[2] is not None else None,
                                "lon": float(row[3]) if len(row) > 3 and row[3] is not None else None,
                                "euclid_distance_m": float(row[4]) if len(row) > 4 and row[4] is not None else 0.0,
                            }
                            for row in candidates
                        ]
                        result["candidate_list"] = candidate_list
                        result["num_candidates"] = len(candidate_list)
                        result["sql_execution_ok"] = True
                        
                        print(f"\nSQL executed successfully. Candidates: {len(candidate_list)}")
                        if candidate_list:
                            print(f"Top 5 candidates:")
                            for idx, cand in enumerate(candidate_list[:5], 1):
                                print(f"  {idx}. {cand.get('name', 'N/A')} ({cand.get('fclass', 'N/A')}) - "
                                      f"dist: {cand.get('euclid_distance_m', 0):.1f}m")
                    else:
                        result["sql_execution_ok"] = True
                        result["num_candidates"] = 0
                        print(f"\nSQL executed successfully. No candidates returned.")
                    
                    # 7. ROUTING (ENABLED with top_k=5) - Using SSSP approach like GT generation
                    if skip_routing:
                        print(f"\n{'='*80}")
                        print("7. ROUTING (SKIPPED)")
                        print(f"{'='*80}")
                        result["routing_failed"] = False
                        result["final_poi"] = []
                        result["top_5_routing_abnormal_pois"] = []
                    elif candidate_list and anchor_lat is not None and anchor_lon is not None:
                        print(f"\n{'='*80}")
                        print("7. ROUTING (top_k=5)")
                        print(f"{'='*80}")
                        print(f"Scenario: {scenario} (mode: {best_mode})")
                        
                        try:
                            import osmnx as ox
                            import math
                            # Ensure routing_module imports resolve
                            if str(SCRIPTS_DIR) not in sys.path:
                                sys.path.insert(0, str(SCRIPTS_DIR))
                            from routing_module.routing_core import map_points_to_nodes
                            from routing_module.sssp import sssp_dijkstra, reconstruct_path
                            
                            # Load graph for abnormal scenario (same as GT generation)
                            GRAPH_CACHE_DIR = BASE / "Data" / "graph-cache"
                            GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                            mode_label = best_mode.lower() if isinstance(best_mode, str) else "drive"
                            
                            # Get city name for graph file (convert spaces to underscores for filename)
                            city_name = city or record.get("city", "Chicago")
                            city_name_for_graph = city_name.replace(" ", "_")
                            
                            # Load city-specific graph
                            if mode_label == "walk":
                                graph_name = f"{city_name_for_graph}_walk_abnormal.graphml"
                                weight_key = "walk_time"
                            else:
                                graph_name = f"{city_name_for_graph}_drive_abnormal.graphml"
                                weight_key = "travel_time"
                            
                            graph_path = GRAPH_CACHE_DIR / graph_name
                            
                            # Fallback to Chicago if city-specific graph not found
                            if not graph_path.exists() and city_name != "Chicago":
                                if mode_label == "walk":
                                    chicago_graph_name = "Chicago_walk_abnormal.graphml"
                                else:
                                    chicago_graph_name = "Chicago_drive_abnormal.graphml"
                                graph_path = GRAPH_CACHE_DIR / chicago_graph_name
                                if graph_path.exists():
                                    print(f"  ⚠️  {city_name_for_graph} graph not found, using Chicago graph as fallback")
                            
                            if not graph_path.exists():
                                print(f"  ⚠️  Graph not found: {graph_name} (tried {city_name} and Chicago)")
                                result["routing_failed"] = True
                            else:
                                G = ox.load_graphml(graph_path)
                                
                                # Sanitize graph weights - ensure all weights are numeric (handle string 'inf' case)
                                for u, v, k, data in G.edges(keys=True, data=True):
                                    weight = data.get(weight_key)
                                    if weight is None:
                                        continue
                                    # Handle string 'inf' case explicitly
                                    if isinstance(weight, str) and weight.lower() in ('inf', 'infinity'):
                                        data[weight_key] = float("inf")
                                    else:
                                        try:
                                            val = float(weight)
                                            if val <= 0 or not math.isfinite(val):
                                                data[weight_key] = float("inf")
                                            else:
                                                data[weight_key] = val
                                        except (ValueError, TypeError):
                                            # If conversion fails, set a default based on length
                                            length = float(data.get("length", 0.0))
                                            if weight_key == "walk_time":
                                                data[weight_key] = float(length / 1.4 if length > 0 else 1.0)
                                            elif weight_key == "travel_time":
                                                data[weight_key] = float(length / 13.4 if length > 0 else 1.0)
                                            else:
                                                data[weight_key] = 1.0
                                
                                # Map points to nodes
                                all_lats = [anchor_lat] + [float(c.get("lat", 0)) for c in candidate_list]
                                all_lons = [anchor_lon] + [float(c.get("lon", 0)) for c in candidate_list]
                                all_nodes = map_points_to_nodes(G, all_lats, all_lons)
                                
                                if not all_nodes or all_nodes[0] is None:
                                    print(f"  ⚠️  Failed to map anchor to graph node")
                                    result["routing_failed"] = True
                                else:
                                    # Convert node IDs to Python int (same as EVClarify GT generation)
                                    src_node = int(all_nodes[0]) if all_nodes[0] is not None else None
                                    if src_node is None:
                                        print(f"  ⚠️  Failed to map anchor to graph node")
                                        result["routing_failed"] = True
                                    else:
                                        cand_nodes = [int(n) for n in all_nodes[1:] if n is not None]
                                        
                                        if not cand_nodes:
                                            print(f"  ⚠️  No candidate nodes found")
                                            result["routing_failed"] = True
                                        else:
                                            # Build node to candidate index mapping
                                            node_to_cand_idx = {}
                                            valid_cand_nodes = []
                                            for idx, node in enumerate(all_nodes[1:]):
                                                if node is not None:
                                                    node_int = int(node)
                                                    node_to_cand_idx[node_int] = idx
                                                    valid_cand_nodes.append(node_int)
                                        
                                        # Get top-k routes using SSSP with correct weight_key (same as GT generation)
                                        # Use local pick_top_k_routes function that accepts weight_key
                                        def pick_top_k_routes_local(G_sub, source_node: int, cand_nodes: List[int], weight_key: str, k_min: int = 1, k_max: int = 5):
                                            """Pick top-K routes using specified weight key (same as generate_true_intent.py)."""
                                            seen = set()
                                            uniq = []
                                            for n in cand_nodes:
                                                if n not in seen:
                                                    seen.add(n)
                                                    uniq.append(n)
                                            distances, paths = sssp_dijkstra(G_sub, int(source_node), weight=weight_key)
                                            results = []
                                            for cn in uniq:
                                                cn_int = int(cn)  # Ensure Python int
                                                if cn_int not in distances:
                                                    continue
                                                tt = float(distances[cn_int])
                                                if not math.isfinite(tt):
                                                    continue
                                                path = reconstruct_path(paths, int(source_node), cn_int)
                                                if not path or len(path) < 2 or tt <= 0.0:
                                                    continue
                                                results.append((cn_int, tt, path))
                                            results.sort(key=lambda x: x[1])
                                            if len(results) < k_min:
                                                return results
                                            return results[: min(k_max, len(results))]
                                        
                                        routes = pick_top_k_routes_local(G, src_node, valid_cand_nodes, weight_key, k_min=1, k_max=TOP_K)
                                        
                                        if not routes:
                                            print(f"  ⚠️  No routes found")
                                            result["routing_failed"] = True
                                            result["final_poi"] = []
                                            result["top_5_routing_abnormal_pois"] = []
                                        else:
                                            # Process routes and compute distances (same as GT generation)
                                            scored_candidates = []
                                            for node, travel_time_s, path in routes:
                                                cand_idx = node_to_cand_idx.get(node)
                                                if cand_idx is not None and cand_idx < len(candidate_list):
                                                    poi = candidate_list[cand_idx]
                                                    # Compute route distance by summing edge lengths (no geometry saved)
                                                    length_m = 0.0
                                                    for i in range(len(path) - 1):
                                                        u, v = path[i], path[i + 1]
                                                        edge_data = G.get_edge_data(u, v)
                                                        if edge_data:
                                                            # Get length from first edge data (multi-edge graph)
                                                            first_edge = next(iter(edge_data.values()))
                                                            length_m += float(first_edge.get("length", 0.0))
                                                    
                                                    scored_candidates.append({
                                                        "poi": poi,
                                                        "cost": travel_time_s,
                                                        "distance": length_m,
                                                    })
                                            
                                            # Sort by cost and format results
                                            scored_candidates.sort(key=lambda x: x["cost"])
                                            
                                            result["final_poi"] = [
                                                format_poi_result({
                                                    "name": item["poi"].get("name") or "NONE",
                                                    "fclass": item["poi"].get("fclass") or best_target,
                                                    "route_cost_s": item["cost"],
                                                    "distance_m": item["distance"],
                                                })
                                                for item in scored_candidates
                                            ]
                                            
                                            # Also save as top_5_routing_abnormal_pois for clarity
                                            result["top_5_routing_abnormal_pois"] = result["final_poi"]
                                            
                                            print(f"\nTop {len(result['final_poi'])} POIs after routing (scenario: {scenario}):")
                                            for idx, poi in enumerate(result["final_poi"], 1):
                                                print(f"  {idx}. {poi.get('poi_name', 'N/A')} ({poi.get('poi_type', 'N/A')}) - "
                                                      f"cost: {poi.get('route_cost_s', 0):.1f}s, dist: {poi.get('distance_m', 0):.1f}m")
                        except Exception as e:
                            result["routing_failed"] = True
                            print(f"  ⚠️  Routing failed: {str(e)}")
                            import traceback
                            traceback.print_exc()
                
                except Exception as e:
                    result["sql_execution_ok"] = False
                    result["num_candidates"] = 0
                    result["sql_error"] = str(e)
                    print(f"\nSQL execution failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
    except Exception as e:
        result["error"] = f"Query processing failed: {str(e)}"
        import traceback
        traceback.print_exc()
    
    return result


# ============================================================================
# Main function
# ============================================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser(description="Unified evaluation script with IG policy")
    ap.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o-mini, gpt-4, qwen2.5-14b-instruct, ministral-8b)")
    ap.add_argument("--max_turns", type=int, default=5, help="Maximum dialogue turns")
    ap.add_argument("--eval_query_csv", type=str, default=None, help=f"Path to eval query CSV file (default: {EVAL_QUERY_CSV})")
    ap.add_argument("--dialogue_file", type=str, default=None, help=f"Path to dialogue JSON file (default: {DIALOGUE_FILE})")
    ap.add_argument("--true_intent_file", type=str, default=None, help=f"Path to true intent JSON file (default: {TRUE_INTENT_FILE})")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: Result/{model})")
    ap.add_argument("--query_id", type=str, default=None, help="Test a specific query ID")
    ap.add_argument("--samples", type=int, default=None, help="Number of queries to run (default: all)")
    ap.add_argument("--enable-routing", action="store_true", default=False, help="Enable routing step (default: False, routing is skipped by default to speed up evaluation)")
    args = ap.parse_args()
    
    # Determine model type
    model_type = get_model_type(args.model)
    
    # Set defaults if not provided
    if args.eval_query_csv is None:
        args.eval_query_csv = str(EVAL_QUERY_CSV)
    if args.out_dir is None:
        model_name_safe = args.model.replace("/", "_").replace("-", "_").replace(".", "_")
        args.out_dir = str(RESULT_DIR / model_name_safe)
    if args.dialogue_file is None:
        args.dialogue_file = str(DIALOGUE_FILE)
    if args.true_intent_file is None:
        args.true_intent_file = str(TRUE_INTENT_FILE)
    
    # Load eval queries from CSV (just query_id and query)
    eval_query_csv = Path(args.eval_query_csv)
    if not eval_query_csv.exists():
        raise SystemExit(f"Eval query CSV file not found: {eval_query_csv}")
    
    eval_queries = []
    with open(eval_query_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eval_queries.append({
                "query_id": row["query_id"],
                "query": row["query"]
            })
    
    # Filter to specific query_id if provided
    if args.query_id:
        query_id_normalized = args.query_id.replace("_", "-")
        eval_queries = [q for q in eval_queries if q.get("query_id", "").replace("_", "-") == query_id_normalized]
        if not eval_queries:
            raise SystemExit(f"Query ID '{args.query_id}' not found in eval queries")
        print(f"Filtered to query_id: {args.query_id} (found {len(eval_queries)} matching query/queries)")
    
    # Load EVClarify files for parameters and GTs
    dialogue_data = {}
    dialogue_file = Path(args.dialogue_file)
    if dialogue_file.exists():
        with open(dialogue_file) as f:
            dialogue_list = json.load(f)
            dialogue_data = {rec["query_id"]: rec for rec in dialogue_list}
    else:
        print(f"Warning: Dialogue file not found: {dialogue_file}")
    
    true_intent_data = {}
    true_intent_file = Path(args.true_intent_file)
    if true_intent_file.exists():
        with open(true_intent_file) as f:
            true_intent_list = json.load(f)
            true_intent_data = {rec["query_id"]: rec for rec in true_intent_list}
    else:
        print(f"Warning: True intent file not found: {true_intent_file}")
    
    # Merge eval queries with EVClarify data
    test_queries = []
    for eval_q in eval_queries:
        query_id = eval_q["query_id"]
        # Get data from EVClarify files
        true_intent_rec = true_intent_data.get(query_id, {})
        # Create record with query from CSV and other data from EVClarify
        record = {
            "query_id": query_id,
            "query": eval_q["query"],
            "city": true_intent_rec.get("city", "Chicago"),
            "true_intent": true_intent_rec.get("true_intent", {}),
        }
        test_queries.append(record)
    
    # Limit samples if specified
    if args.samples:
        test_queries = test_queries[:args.samples]
    
    # Determine output files
    result_dir = Path(args.out_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    model_name_clean = args.model.replace("-", "_").replace(".", "_").replace("/", "_")
    json_file = result_dir / f"eval_{model_name_clean}.json"
    csv_file = result_dir / f"eval_{model_name_clean}.csv"
    candidates_file = result_dir / f"eval_{model_name_clean}_candidates.json"
    
    # Load existing results if resuming
    results: List[Dict[str, Any]] = []
    processed_query_ids = set()
    
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                existing_results = json.load(f)
                results = existing_results
                processed_query_ids = {r.get("query_id") for r in existing_results if r.get("query_id")}
                print(f"Loaded {len(existing_results)} existing results from {json_file}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    print("=" * 80)
    print("EVALUATION (IG Policy)")
    print("=" * 80)
    print(f"Model: {args.model} ({model_type})")
    print(f"Total queries: {len(test_queries)}")
    print(f"Already processed: {len(processed_query_ids)}")
    print(f"Remaining: {len(test_queries) - len(processed_query_ids)}")
    print(f"Top-K (routing): {TOP_K}")
    print(f"Max turns: {args.max_turns}")
    
    # Load min_prob values from .txt file (once at startup)
    target_min_prob, mode_min_prob = load_min_prob_values()
    print()
    
    # Get client based on model
    try:
        client = get_client_for_model(args.model)
    except Exception as e:
        raise SystemExit(f"Failed to initialize client for model '{args.model}': {e}")
    
    scenario = "abnormal"
    
    for i, record in enumerate(test_queries, 1):
        query_id = record.get("query_id", f"UNKNOWN_{i}")
        
        # Skip if already processed
        if query_id in processed_query_ids:
            print(f"[{i}/{len(test_queries)}] Skipping {query_id} (already processed)")
            continue
        
        query = record.get("query", "")
        dialogue_record = dialogue_data.get(query_id)
        
        print(f"[{i}/{len(test_queries)}] Processing {query_id}...")
        print(f"\n{'='*80}")
        print(f"Query ID: {query_id}")
        print(f"Query: {query}")
        
        # Detect city from record
        city_from_record = record.get("city", None)
        
        try:
            result = run_one_query(
                client,
                query_id,
                query,
                record,
                dialogue_record,
                args.model,
                model_type,
                args.max_turns,
                scenario,
                target_min_prob,
                mode_min_prob,
                city=city_from_record,
                skip_routing=not args.enable_routing,
            )
            
            results.append(result)
            
            # Save after each query (incremental)
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save candidates separately
            all_candidates = {}
            for r in results:
                if r.get("candidate_list"):
                    all_candidates[r.get("query_id")] = r.get("candidate_list")
            with open(candidates_file, 'w') as f:
                json.dump(all_candidates, f, indent=2, ensure_ascii=False)
            
            # Update CSV incrementally
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "query_id", "query", "initial_parsed_intent", "identified_entropies",
                    "pred_ambiguity", "anchor_cluster_count", "threshold_cluster_count",
                    "dialogue", "num_turns", "answers", "final_intent", "generated_sql",
                    "sql_execution_ok", "num_candidates",
                ])
                writer.writeheader()
                for r in results:
                    row = {
                        "query_id": r.get("query_id"),
                        "query": r.get("query", ""),
                        "initial_parsed_intent": json.dumps(r.get("initial_parsed_intent", {}), ensure_ascii=False),
                        "identified_entropies": json.dumps(r.get("identified_entropies", []), ensure_ascii=False),
                        "pred_ambiguity": r.get("pred_ambiguity", False),
                        "anchor_cluster_count": r.get("anchor_cluster_count", 0),
                        "threshold_cluster_count": r.get("threshold_cluster_count", 0),
                        "dialogue": json.dumps(r.get("dialogue", []), ensure_ascii=False),
                        "num_turns": r.get("num_turns", 0),
                        "answers": json.dumps(r.get("answers", []), ensure_ascii=False),
                        "final_intent": json.dumps(r.get("final_intent", {}), ensure_ascii=False),
                        "generated_sql": r.get("generated_sql", ""),
                        "sql_execution_ok": r.get("sql_execution_ok", False),
                        "num_candidates": r.get("num_candidates", 0),
                    }
                    writer.writerow(row)
            
        except Exception as e:
            print(f"\n✗ Error processing {query_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            error_result = {
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "initial_parsed_intent": None,
                "identified_entropies": [],
                "pred_ambiguity": False,
                "anchor_cluster_count": 0,
                "threshold_cluster_count": 0,
                "cluster_list": [],
                "dialogue": [],
                "num_turns": 0,
                "answers": [],
                "final_intent": None,
                "generated_sql": None,
                "sql_execution_ok": False,
                "num_candidates": 0,
                "candidate_list": [],
            }
            results.append(error_result)
            
            # Save error result
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Final save
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    all_candidates = {}
    for r in results:
        if r.get("candidate_list"):
            all_candidates[r.get("query_id")] = r.get("candidate_list")
    with open(candidates_file, 'w') as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)
    
    # Final CSV save
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "query", "initial_parsed_intent", "identified_entropies",
            "pred_ambiguity", "anchor_cluster_count", "threshold_cluster_count",
            "dialogue", "num_turns", "answers", "final_intent", "generated_sql",
            "sql_execution_ok", "num_candidates",
        ])
        writer.writeheader()
        for r in results:
            row = {
                "query_id": r.get("query_id"),
                "query": r.get("query", ""),
                "initial_parsed_intent": json.dumps(r.get("initial_parsed_intent", {}), ensure_ascii=False),
                "identified_entropies": json.dumps(r.get("identified_entropies", []), ensure_ascii=False),
                "pred_ambiguity": r.get("pred_ambiguity", False),
                "anchor_cluster_count": r.get("anchor_cluster_count", 0),
                "threshold_cluster_count": r.get("threshold_cluster_count", 0),
                "dialogue": json.dumps(r.get("dialogue", []), ensure_ascii=False),
                "num_turns": r.get("num_turns", 0),
                "answers": json.dumps(r.get("answers", []), ensure_ascii=False),
                "final_intent": json.dumps(r.get("final_intent", {}), ensure_ascii=False),
                "generated_sql": r.get("generated_sql", ""),
                "sql_execution_ok": r.get("sql_execution_ok", False),
                "num_candidates": r.get("num_candidates", 0),
            }
            writer.writerow(row)
    
    print()
    print("=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print(f"Results saved to:")
    print(f"  Main JSON: {json_file}")
    print(f"  Candidates JSON: {candidates_file}")
    print(f"  CSV: {csv_file}")
    print(f"Total queries: {len(results)}")
    print(f"Successful: {sum(1 for r in results if not r.get('error'))}")
    print(f"Failed: {sum(1 for r in results if r.get('error'))}")


if __name__ == "__main__":
    main()

