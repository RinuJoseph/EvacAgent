
"""
LLM on for clarification simulation:
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIError, RateLimitError

import uq


def _extract_json(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = "\n".join([ln for ln in s.splitlines() if not ln.strip().startswith("```")]).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return json.loads(s[start : end + 1])


def assistant_question_llm(
    client: OpenAI,
    slot: str,
    options: List[Dict[str, Any]],  # Not used, kept for compatibility
    *,
    model: str = "gpt-4o-mini",
    max_retries: int = 6,
) -> Dict[str, Any]:
    """
    Generate a clarification question for the given slot.
    Options are NOT provided to the LLM - only the slot name.
    Returns: {"question": "..."} only.
    """
    import prompts
    payload = {"slot": slot}
    prompt = prompts.ASSISTANT_QUESTION_PROMPT.format(
        payload=json.dumps(payload, ensure_ascii=False)
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=80,
            )
            # Handle Mistral response format (dict) vs OpenAI format (object)
            if isinstance(resp, dict):
                content = resp.get("choices", [{}])[0].get("message", {}).get("content") or ""
            else:
                content = resp.choices[0].message.content or ""
            return _extract_json(content)
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            last_err = e
            time.sleep(min(2.0**attempt, 10.0))
    raise RuntimeError(f"assistant_question_llm failed after {max_retries} retries: {last_err}")


def user_answer_llm(
    client: OpenAI,
    slot: str,
    options: List[Dict[str, Any]],
    true_intent: Dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
    max_retries: int = 6,
) -> Dict[str, Any]:
    """
    Returns: {"choice_id": "<id>"} where id must be one of options[*].id.
    For anchor questions, uses true anchor coordinates from true_intent directly (not cluster matching).
    """
    import prompts
    # For anchor questions, provide true anchor coordinates and option coordinates for matching
    if slot == "anchor":
        # Include true anchor coordinates from GT
        true_anchor = {
            "lat": float(true_intent.get("anchor_lat", 0.0)),
            "lon": float(true_intent.get("anchor_lon", 0.0))
        }
        # Include option coordinates for matching
        options_with_coords = []
        for opt in options:
            opt_copy = dict(opt)
            if "lat" in opt and "lon" in opt:
                opt_copy["coordinates"] = {"lat": opt["lat"], "lon": opt["lon"]}
            options_with_coords.append(opt_copy)
        payload = {
            "slot": slot,
            "true_anchor_location": true_anchor,  # True anchor from GT
            "options": options_with_coords,
            "true_intent": true_intent
        }
    else:
        payload = {"slot": slot, "options": options, "true_intent": true_intent}
    
    prompt = prompts.USER_ANSWER_PROMPT.format(
        payload=json.dumps(payload, ensure_ascii=False)
    )
    last_err: Exception | None = None
    valid = {str(o.get("id")) for o in options}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=30,
            )
            # Handle Mistral response format (dict) vs OpenAI format (object)
            if isinstance(resp, dict):
                content = resp.get("choices", [{}])[0].get("message", {}).get("content") or ""
            else:
                content = resp.choices[0].message.content or ""
            obj = _extract_json(content)
            cid = str(obj.get("choice_id") or "").strip()
            if cid not in valid:
                raise ValueError(f"LLM chose invalid id: {cid!r}")
            return {"choice_id": cid}
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError, ValueError) as e:
            last_err = e
            time.sleep(min(2.0**attempt, 10.0))
    raise RuntimeError(f"user_answer_llm failed after {max_retries} retries: {last_err}")


def oracle_answer(
    question_type: str,
    options: List[str],
    true_intent: Dict[str, Any],
    anchor_clusters: List[Dict[str, Any]],
    threshold_clusters: List[Dict[str, Any]],
) -> str:
    """
    Simulate user answer from ground truth.
    Returns the TRUE INTENT VALUE directly (not cluster ID):
    - For AskAnchor: returns coordinates string "lat,lon"
    - For AskThreshold: returns threshold value string
    - For AskTarget/AskMode: returns the true label
    """
    if question_type == "AskTarget":
        gt_target = str(true_intent.get("target_label", ""))
        # ALWAYS return GT target, even if not in options (FULL FIX)
        if gt_target:
            return gt_target
        # Only fallback to options[0] if GT target is empty/missing
        return options[0] if options else ""
    
    if question_type == "AskMode":
        gt_mode = str(true_intent.get("mode_label", ""))
        if gt_mode in options:
            return gt_mode
        return options[0] if options else ""
    
    if question_type == "AskAnchor":
        # Return true anchor coordinates directly (not cluster ID)
        gt_lat = true_intent.get("anchor_lat")
        gt_lon = true_intent.get("anchor_lon")
        
        if gt_lat is not None and gt_lon is not None:
            # Return coordinates string directly
            return f"{float(gt_lat)},{float(gt_lon)}"
        
        # Fallback: use first cluster's coordinates if GT not available
        if anchor_clusters and options:
            first_cluster = next((c for c in anchor_clusters if c["id"] == options[0]), None)
            if first_cluster:
                return f"{first_cluster['lat']},{first_cluster['lon']}"
        
        return ""
    
    if question_type == "AskThreshold":
        # Return true threshold value directly (not cluster ID)
        gt_threshold_m = true_intent.get("threshold_m")
        
        if gt_threshold_m is not None:
            # Return threshold value string directly
            return str(float(gt_threshold_m))
        
        # Fallback: use first numeric cluster's value if GT is null
        if threshold_clusters and options:
            for opt_id in options:
                for th in threshold_clusters:
                    if th["id"] == opt_id and th["kind"] == "meters" and th["threshold_m"] is not None:
                        return str(th["threshold_m"])
            # If no numeric, return "null" or empty
            for opt_id in options:
                for th in threshold_clusters:
                    if th["id"] == opt_id and th["kind"] == "null":
                        return "null"
        
        return ""
    
    # Fallback for unknown question types
    return options[0] if options else ""


