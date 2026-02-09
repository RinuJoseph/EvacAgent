
"""
Information Gain (IG) based question selection policy.

"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import uq


def compute_entropy(probs: Dict[str, float]) -> float:
    """
    Compute Shannon entropy in bits: H(P) = -sum(p * log2(p)) for p > 0.
    
    Args:
        probs: Dictionary mapping keys to probabilities (should sum to ~1.0)
    
    Returns:
        Entropy in bits (0 if all mass on one item, >0 if spread)
    """
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)


def compute_ig_scores(
    anchor_probs: Dict[str, float],
    target_probs: Dict[str, float],
    threshold_probs: Dict[str, float],
    mode_probs: Optional[Dict[str, float]] = None,
    lambda_anchor: float = 0.0,  # deprecated, kept for compatibility (not used)
    lambda_target: float = 0.0,  # deprecated, kept for compatibility (not used)
    lambda_threshold: float = 0.0,  # deprecated, kept for compatibility (not used)
    lambda_mode: float = 0.0,  # deprecated, kept for compatibility (not used)
) -> Dict[str, Dict[str, Any]]:
    """
    Compute information gain scores for each question type.
    Uses entropy directly as the score (lambda removed).
    
    Args:
        anchor_probs: Probability distribution over anchor clusters
        target_probs: Probability distribution over target types
        threshold_probs: Probability distribution over threshold clusters
        mode_probs: Optional probability distribution over modes
        lambda_*: Deprecated, kept for compatibility (not used)
    
    Returns:
        Dictionary with scores for each question type:
        {
            "AskAnchor": {"ig": float, "score": float, "entropy": float},
            "AskTarget": {...},
            "AskThreshold": {...},
            "AskMode": {...} (if mode_probs provided)
        }
    """
    scores: Dict[str, Dict[str, Any]] = {}
    
    # AskAnchor
    if len(anchor_probs) > 1:
        h_anchor = compute_entropy(anchor_probs)
        scores["AskAnchor"] = {
            "ig": h_anchor,
            "score": h_anchor,  # Use entropy directly (lambda removed)
            "entropy": h_anchor,
        }
    
    # AskTarget
    if len(target_probs) > 1:
        h_target = compute_entropy(target_probs)
        scores["AskTarget"] = {
            "ig": h_target,
            "score": h_target,  # Use entropy directly (lambda removed)
            "entropy": h_target,
        }
    
    # AskThreshold (if clusters > 1, consider it - identified_slots logic handles filtering)
    if len(threshold_probs) > 1:
        h_threshold = compute_entropy(threshold_probs)
        # Include threshold if clusters > 1 (identified_slots will determine if it's truly ambiguous)
        # The stopping criteria will handle whether to actually ask
        scores["AskThreshold"] = {
            "ig": h_threshold,
            "score": h_threshold,  # Use entropy directly (lambda removed)
            "entropy": h_threshold,
        }
    
    # AskMode (if ambiguous)
    if mode_probs and len(mode_probs) > 1:
        h_mode = compute_entropy(mode_probs)
        scores["AskMode"] = {
            "ig": h_mode,
            "score": h_mode,  # Use entropy directly (lambda removed)
            "entropy": h_mode,
        }
    
    return scores


def select_question(
    anchor_probs: Dict[str, float],
    target_probs: Dict[str, float],
    threshold_probs: Dict[str, float],
    mode_probs: Optional[Dict[str, float]] = None,
    lambda_anchor: float = 0.0,  # deprecated, kept for compatibility (not used)
    lambda_target: float = 0.0,  # deprecated, kept for compatibility (not used)
    lambda_threshold: float = 0.0,  # deprecated, kept for compatibility (not used)
    lambda_mode: float = 0.0,  # deprecated, kept for compatibility (not used)
    min_score: float = 0.0,
    tau_bits: float = 0.1,  # Adjusted to be less aggressive - only stop if entropy is very low
    p_stop: float = 0.95,  # Adjusted to be more strict - require very high confidence to stop
) -> Dict[str, Any]:
    """
    Select next question using information gain (entropy), with stopping rules.
    Lambda (question cost) has been removed - uses entropy directly.
    
    Args:
        anchor_probs: Probability distribution over anchor clusters
        target_probs: Probability distribution over target types
        threshold_probs: Probability distribution over threshold clusters
        mode_probs: Optional probability distribution over modes
        lambda_*: Deprecated, kept for compatibility (not used)
        min_score: Minimum score required to ask a question
        tau_bits: Stop if max entropy < tau_bits
        p_stop: Stop if max prob > p_stop for all marginals
    
    Returns:
        {
            "action": "AskQuestion" | "NoQuestion",
            "question_type": str | None,
            "reason": str,
            "scores": Dict[str, Dict],
            "entropies": Dict[str, float],
        }
    """
    # Compute entropies
    h_anchor = compute_entropy(anchor_probs) if len(anchor_probs) > 1 else 0.0
    h_target = compute_entropy(target_probs) if len(target_probs) > 1 else 0.0
    h_threshold = compute_entropy(threshold_probs) if len(threshold_probs) > 1 else 0.0
    h_mode = compute_entropy(mode_probs) if mode_probs and len(mode_probs) > 1 else 0.0
    
    entropies = {
        "anchor": h_anchor,
        "target": h_target,
        "threshold": h_threshold,
        "mode": h_mode,
    }
    
    # Check stopping rules
    # Only stop if max entropy is very low (adjusted to be less aggressive)
    max_entropy = max([h_anchor, h_target, h_threshold, h_mode])
    if max_entropy < tau_bits:
        return {
            "action": "NoQuestion",
            "question_type": None,
            "reason": f"Max entropy {max_entropy:.3f} < threshold {tau_bits}",
            "scores": {},
            "entropies": entropies,
        }
    
    # Check max-prob stopping rule
    # Adjusted: Only stop if ALL slots have very high confidence (increased threshold)
    # This ensures identified ambiguous slots (entropy >= 1.0 or clusters > 1) are still considered
    max_p_anchor = max(anchor_probs.values()) if anchor_probs else 1.0
    max_p_target = max(target_probs.values()) if target_probs else 1.0
    max_p_threshold = max(threshold_probs.values()) if threshold_probs else 1.0
    max_p_mode = max(mode_probs.values()) if mode_probs else 1.0
    
    # Only stop if all slots have very high confidence (>= 0.95) AND low entropy
    # This prevents stopping when there are identified ambiguous slots
    if (max_p_anchor >= 0.95 and max_p_target >= 0.95 and 
        max_p_threshold >= 0.95 and (not mode_probs or max_p_mode >= 0.95) and
        max_entropy < 0.2):  # Also require low entropy
        return {
            "action": "NoQuestion",
            "question_type": None,
            "reason": f"All max probs >= 0.95 and max entropy < 0.2",
            "scores": {},
            "entropies": entropies,
        }
    
    # Compute IG scores
    scores = compute_ig_scores(
        anchor_probs, target_probs, threshold_probs, mode_probs,
        lambda_anchor, lambda_target, lambda_threshold, lambda_mode
    )
    
    if not scores:
        return {
            "action": "NoQuestion",
            "question_type": None,
            "reason": "No ambiguous slots",
            "scores": scores,
            "entropies": entropies,
        }
    
    # Select question with highest score
    best_question = max(scores.keys(), key=lambda q: scores[q]["score"])
    best_score = scores[best_question]["score"]
    
    if best_score < min_score:
        return {
            "action": "NoQuestion",
            "question_type": None,
            "reason": f"Best score {best_score:.2f} < min {min_score}",
            "scores": scores,
            "entropies": entropies,
        }
    
    return {
        "action": "AskQuestion",
        "question_type": best_question,
        "reason": None,
        "scores": scores,
        "entropies": entropies,
        "best_score": best_score,
    }


def update_marginal_after_answer(
    current_probs: Dict[str, float],
    answer: str,
) -> Dict[str, float]:
    """
    Update marginal distribution after user answers a question.
    Collapses to point mass on the answer.
    
    Args:
        current_probs: Current probability distribution
        answer: User's answer (key in the distribution)
    
    Returns:
        Updated distribution with point mass on answer
    """
    if answer in current_probs:
        return {answer: 1.0}
    else:
        # If answer not in distribution, add it with probability 1.0
        # This handles cases where the oracle answer is correct but wasn't in initial distribution
        return {answer: 1.0}


def build_marginals_from_belief_debug_info(
    debug_info: Dict[str, Any],
    parsed: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Optional[Dict[str, float]]]:
    """
    Build marginal distributions from belief debug_info and parsed query.
    
    This extracts marginals without building joint worlds.
    
    Args:
        debug_info: Debug info from build_belief (contains anchor_clusters, threshold_clusters)
        parsed: Parsed query with target_poi_type and navigation_mode probabilities
    
    Returns:
        (anchor_probs, target_probs, threshold_probs, mode_probs)
    """
    # Anchor marginals from clusters
    anchor_clusters = debug_info.get("anchor_clusters", []) or []
    anchor_probs: Dict[str, float] = {}
    for cluster in anchor_clusters:
        cluster_id = cluster.get("id", "")
        mass = float(cluster.get("mass", 0.0))
        if mass > 0 and cluster_id:
            anchor_probs[cluster_id] = mass
    
    # Normalize anchor probs
    total_anchor = sum(anchor_probs.values())
    if total_anchor > 0:
        anchor_probs = {k: v / total_anchor for k, v in anchor_probs.items()}
    
    # Target marginals from parsed query (with CP filtering)
    target_probs_raw = parsed.get("target_poi_type", {}) or {}
    target_probs_full = uq.complete_and_normalize(target_probs_raw, uq.TARGET_TYPES)
    cp_min_t = debug_info.get("cp_min_prob_target", uq.DEFAULT_TARGET_MIN_PROB)
    target_set = list(uq.heuristic_set_from_probs(target_probs_full, cp_min_t))
    target_probs: Dict[str, float] = {}
    for target_type in target_set:
        p = float(target_probs_full.get(target_type, 0.0))
        if p > 0:
            target_probs[target_type] = p
    
    # Normalize target probs
    total_target = sum(target_probs.values())
    if total_target > 0:
        target_probs = {k: v / total_target for k, v in target_probs.items()}
    
    # Threshold marginals from clusters
    threshold_clusters = debug_info.get("threshold_clusters", []) or []
    threshold_probs: Dict[str, float] = {}
    for cluster in threshold_clusters:
        cluster_id = cluster.get("id", "")
        mass = float(cluster.get("mass", 0.0))
        if mass > 0 and cluster_id:
            threshold_probs[cluster_id] = mass
    
    # Normalize threshold probs
    total_threshold = sum(threshold_probs.values())
    if total_threshold > 0:
        threshold_probs = {k: v / total_threshold for k, v in threshold_probs.items()}
    
    # Mode marginals from parsed query (with CP filtering)
    mode_probs_raw = parsed.get("navigation_mode", {}) or {}
    mode_probs_full = uq.complete_and_normalize(mode_probs_raw, uq.NAV_MODES)
    cp_min_m = debug_info.get("cp_min_prob_mode", uq.DEFAULT_MODE_MIN_PROB)
    mode_set = list(uq.heuristic_set_from_probs(mode_probs_full, cp_min_m))
    mode_probs: Optional[Dict[str, float]] = {}
    for mode in mode_set:
        p = float(mode_probs_full.get(mode, 0.0))
        if p > 0:
            mode_probs[mode] = p
    
    # Normalize mode probs
    total_mode = sum(mode_probs.values()) if mode_probs else 0
    if total_mode > 0:
        mode_probs = {k: v / total_mode for k, v in mode_probs.items()}
    else:
        mode_probs = None
    
    return anchor_probs, target_probs, threshold_probs, mode_probs

