
"""
Evaluation metrics for dialogue system.
Computes task success rate, conversation turns, ambiguity detection F1, and final answer accuracy.
"""

import json
import math
import re
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict

# Import uq module for anchor_candidates_from_db (same directory)
try:
    import uq
except ImportError:
    # Fallback: try adding method directory to path
    try:
        import paths
        method_path = paths.BASE / "scripts" / "method"
    except ImportError:
        BASE_PATH = Path("/workspace/storage/DSPP-CH/EvacAgent")
        method_path = BASE_PATH / "scripts" / "method"
    if str(method_path) not in sys.path:
        sys.path.insert(0, str(method_path))
    import uq

# Import paths for consistent BASE path
try:
    import paths
    DEFAULT_BASE = paths.BASE  # EvacAgent directory
except ImportError:
    DEFAULT_BASE = Path("/workspace/storage/DSPP-CH/EvacAgent")


def detect_city_from_sql(sql: Optional[str]) -> Optional[str]:
    """Detect city abbreviation from SQL table name (e.g., CH_POI -> CH, HO_POI -> HO)."""
    if not sql:
        return None
    # Match patterns like FROM CH_POI, FROM HO_POI, etc.
    match = re.search(r'FROM\s+([A-Z]{2})_POI', sql, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def normalize_query_text(text: str) -> str:
    """Normalize query text for matching (lowercase, strip punctuation, normalize whitespace)."""
    if not text:
        return ""
    # Lowercase, remove extra whitespace
    text = " ".join(text.lower().split())
    # Remove common punctuation that might differ
    text = text.replace(",", "").replace("?", "").replace(".", "").replace("!", "")
    return text.strip()


def build_ea_to_city_query_mapping(base_path: Path = None) -> Dict[str, str]:
    """Build mapping from EA-xxx query IDs to city-specific query IDs by matching query text.
    
    Args:
        base_path: Base path. Defaults to /workspace/storage/DSPP-CH.
    
    Returns:
        Dict mapping EA-xxx -> city-specific query_id (e.g., "EA-000" -> "HO_001")
    """
    if base_path is None:
        base_path = DEFAULT_BASE
    
    # Load test_queries.json to get EA-xxx -> city mapping
    test_queries_file = base_path / "Final-Data" / "test_queries.json"
    if not test_queries_file.exists():
        print(f"Warning: test_queries.json not found at {test_queries_file}")
        return {}
    
    with open(test_queries_file, 'r') as f:
        test_queries = json.load(f)
    
    # Build EA-xxx -> (city, query_text) mapping
    ea_to_city_query = {}
    for record in test_queries:
        ea_id = record.get("query_id")
        city = record.get("city")
        query_text = record.get("query", "")
        if ea_id and city and query_text:
            ea_to_city_query[ea_id] = {
                "city": city,
                "query_text": normalize_query_text(query_text),
            }
    
    # Load city query files and build city-specific query_id -> normalized query_text mapping
    city_abbrs = ["CH", "HO", "NY", "SA", "NO", "MI"]
    city_name_to_abbr = {
        "Chicago": "CH",
        "Houston": "HO",
        "New York": "NY",
        "San Antonio": "SA",
        "New Orleans": "NO",
        "Miami": "MI",
    }
    
    city_query_lookup = {}  # city_abbr -> {normalized_query -> city_query_id}
    data_path = base_path / "Data"
    
    for city_name, abbr in city_name_to_abbr.items():
        query_file = data_path / f"GT_{abbr}" / f"{abbr}_query.json"
        if query_file.exists():
            try:
                with open(query_file, 'r') as f:
                    city_queries = json.load(f)
                city_query_lookup[abbr] = {}
                for record in city_queries:
                    city_query_id = record.get("query_id")
                    city_query_text = record.get("query", "")
                    if city_query_id and city_query_text:
                        normalized = normalize_query_text(city_query_text)
                        city_query_lookup[abbr][normalized] = city_query_id
            except Exception as e:
                print(f"Warning: Failed to load {query_file}: {e}")
    
    # Match EA-xxx to city-specific query_id by query text
    ea_to_city_id_map = {}
    matched = 0
    for ea_id, info in ea_to_city_query.items():
        city = info["city"]
        normalized_query = info["query_text"]
        city_abbr = city_name_to_abbr.get(city)
        
        if city_abbr and city_abbr in city_query_lookup:
            if normalized_query in city_query_lookup[city_abbr]:
                city_query_id = city_query_lookup[city_abbr][normalized_query]
                ea_to_city_id_map[ea_id] = city_query_id
                matched += 1
    
    print(f"Matched {matched}/{len(ea_to_city_query)} EA queries to city-specific query IDs by query text")
    return ea_to_city_id_map


def load_test_gt_poi_file(base_path: Path = None) -> Dict[str, Dict[str, Any]]:
    """Load unified test GT POI file (Final-Data/test_gt_poi.json) with EA-xxx IDs.
    
    Args:
        base_path: Base path. Defaults to /workspace/storage/DSPP-CH.
    
    Returns:
        Dict mapping EA-xxx query_id -> GT POI record
    """
    if base_path is None:
        base_path = DEFAULT_BASE
    
    test_gt_file = base_path / "Final-Data" / "test_gt_poi.json"
    if not test_gt_file.exists():
        return {}
    
    try:
        with open(test_gt_file, 'r') as f:
            test_gt_data = json.load(f)
        gt_poi_lookup = {}
        for item in test_gt_data:
            query_id = item.get("query_id")
            if query_id:
                # Store with the EA-xxx query_id as key
                gt_poi_lookup[query_id] = item
        return gt_poi_lookup
    except Exception as e:
        print(f"Warning: Failed to load {test_gt_file}: {e}")
        return {}


def load_city_gt_poi_files(base_path: Path = None) -> Dict[str, Dict[str, Any]]:
    """Load all city-wise GT POI abnormal files and return merged lookup by query_id.
    
    Args:
        base_path: Base path to Data directory. Defaults to /workspace/storage/DSPP-CH/Data.
    
    Returns:
        Dict mapping query_id -> GT POI record
    """
    if base_path is None:
        base_path = Path("/workspace/storage/DSPP-CH/Data")
    
    gt_poi_lookup = {}
    city_abbrs = ["CH", "HO", "NY", "SA", "NO", "MI"]
    
    for abbr in city_abbrs:
        gt_file = base_path / f"GT_{abbr}" / f"{abbr}_gt_poi_abnormal.json"
        if gt_file.exists():
            try:
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                for item in gt_data:
                    query_id = item.get("query_id")
                    if query_id:
                        gt_poi_lookup[query_id] = item
            except Exception as e:
                print(f"Warning: Failed to load {gt_file}: {e}")
    
    return gt_poi_lookup


def load_eval_results(eval_json_path: Path, gt_json_path: Path = None, gt_poi_path: Path = None, auto_load_city_gt: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load evaluation results and GT data.
    
    Args:
        eval_json_path: Path to evaluation results JSON
        gt_json_path: Optional path to GT metadata JSON (gt_num_turns, gt_ambiguity_types). 
                      If None, extracts GT metadata from eval_results.
        gt_poi_path: Path to GT POI JSON (GT_Chicago_POI.json). If None and auto_load_city_gt=True,
                     automatically loads city-wise GT files based on SQL table names.
        auto_load_city_gt: If True and gt_poi_path is None, auto-load city GT files.
    
    Returns:
        Tuple of (eval_results, gt_metadata_lookup, gt_poi_lookup)
    """
    with open(eval_json_path, 'r') as f:
        eval_results = json.load(f)
    
    # Try to load candidates from separate candidates file if generated_sql_candidate_list is missing
    # Try multiple possible candidates file names
    candidates_file = None
    possible_names = [
        eval_json_path.parent / f"{eval_json_path.stem}_candidates.json",
        eval_json_path.parent / f"{eval_json_path.stem.replace('_results', '')}_candidates.json",
        eval_json_path.parent / f"{eval_json_path.stem.replace('_results', '')}candidates.json",
    ]
    for name in possible_names:
        if name.exists():
            candidates_file = name
            break
    
    if candidates_file and candidates_file.exists() and not any(r.get("generated_sql_candidate_list") for r in eval_results):
        try:
            with open(candidates_file, 'r', encoding='utf-8') as f:
                candidates_data = json.load(f)
            
            # Add candidates to results
            for result in eval_results:
                query_id = result.get("query_id")
                if query_id and query_id in candidates_data:
                    # Candidates file can have different structures
                    cands = candidates_data[query_id]
                    if isinstance(cands, list):
                        # Direct list of candidates - normalize field names
                        normalized_cands = []
                        for c in cands:
                            norm_c = {}
                            # Normalize field names: lon/long, distance_m/euclid_distance_m
                            norm_c["name"] = c.get("name")
                            norm_c["fclass"] = c.get("fclass")
                            norm_c["lat"] = c.get("lat")
                            norm_c["lon"] = c.get("lon") or c.get("long")
                            norm_c["euclid_distance_m"] = c.get("euclid_distance_m") or c.get("distance_m")
                            normalized_cands.append(norm_c)
                        result["generated_sql_candidate_list"] = normalized_cands
                    elif isinstance(cands, dict):
                        # Check for common field names
                        candidates_list = None
                        if "sql_candidates" in cands:
                            candidates_list = cands["sql_candidates"]
                        elif "candidates" in cands:
                            candidates_list = cands["candidates"]
                        elif "candidate_list" in cands:
                            candidates_list = cands["candidate_list"]
                        
                        # Normalize field names if we found a list
                        if candidates_list and isinstance(candidates_list, list):
                            normalized_cands = []
                            for c in candidates_list:
                                norm_c = {}
                                # Normalize field names: lon/long, distance_m/euclid_distance_m
                                norm_c["name"] = c.get("name")
                                norm_c["fclass"] = c.get("fclass")
                                norm_c["lat"] = c.get("lat")
                                norm_c["lon"] = c.get("lon") or c.get("long")
                                norm_c["euclid_distance_m"] = c.get("euclid_distance_m") or c.get("distance_m") or (c.get("distance") * 1000 if c.get("distance") is not None else None)
                                normalized_cands.append(norm_c)
                            result["generated_sql_candidate_list"] = normalized_cands
                        else:
                            result["generated_sql_candidate_list"] = candidates_list
        except Exception as e:
            print(f"Warning: Could not load candidates from {candidates_file}: {e}")
    
    # Load GT metadata from external file if provided, otherwise try to load from standard location
    gt_metadata_lookup = {}
    if gt_json_path and gt_json_path.exists():
        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f)
        # Create GT metadata lookup by query_id
        gt_metadata_lookup = {item["query_id"]: item for item in gt_data}
    else:
        # Try to load from standard location
        standard_gt_file = Path("/workspace/storage/DSPP-CH/Final-Data/Full-Data/EVClarify_queries.json")
        if standard_gt_file.exists():
            try:
                with open(standard_gt_file, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                gt_metadata_lookup = {item["query_id"]: item for item in gt_data}
            except Exception:
                pass
        
        # If still empty, extract GT metadata from eval_results (fallback)
        if not gt_metadata_lookup:
            for result in eval_results:
                query_id = result.get("query_id")
                if query_id:
                    gt_metadata_lookup[query_id] = {
                        "query_id": query_id,
                        "gt_num_turns": result.get("gt_num_turns"),
                        "gt_ambiguity_types": result.get("gt_ambiguity_types", []),
                        "gt_is_ambiguous": result.get("gt_is_ambiguous", False),
                    }
    
    # Load GT POI data
    gt_poi_lookup = {}
    
    if gt_poi_path and gt_poi_path.exists():
        # Explicit GT POI file provided
        with open(gt_poi_path, 'r') as f:
            gt_poi_data = json.load(f)
        # Create lookup by query_id
        for item in gt_poi_data:
            query_id = item.get("query_id")
            if query_id:
                gt_poi_lookup[query_id] = item
    elif auto_load_city_gt:
        # Check if eval results use EA-xxx IDs
        sample_ids = [r.get("query_id", "") for r in eval_results[:10] if r.get("query_id")]
        uses_ea_ids = any(qid.startswith("EA-") for qid in sample_ids if qid)
        
        if uses_ea_ids:
            # Use unified test GT POI file (Final-Data/test_gt_poi.json)
            gt_poi_lookup = load_test_gt_poi_file()
            print(f"Loaded test GT POI file: {len(gt_poi_lookup)} EA query records found")
        else:
            # Direct lookup (city-specific IDs) - load city-wise GT files
            gt_poi_lookup = load_city_gt_poi_files()
            print(f"Auto-loaded city GT POI files: {len(gt_poi_lookup)} query records found")
    
    # Merge GT metadata into eval results (only if not already present)
    for result in eval_results:
        query_id = result.get("query_id")
        if query_id in gt_metadata_lookup:
            gt_record = gt_metadata_lookup[query_id]
            if "gt_num_turns" not in result:
                result["gt_num_turns"] = gt_record.get("gt_num_turns")
            if "gt_ambiguity_types" not in result:
                result["gt_ambiguity_types"] = gt_record.get("gt_ambiguity_types", [])
            if "gt_is_ambiguous" not in result:
                result["gt_is_ambiguous"] = gt_record.get("gt_is_ambiguous", False)
    
    return eval_results, gt_metadata_lookup, gt_poi_lookup


def compute_task_success_rate(eval_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute task success rate: queries executed without pipeline errors.
    
    A query is successful if:
    - SQL generation did not fail
    - Routing did not fail
    - No error occurred
    
    Note: Success does NOT depend on whether POIs were returned (zero candidates
    is still considered successful if the pipeline ran without errors).
    
    Supports both old format (sql_failed, error) and new format (sql_execution_ok, sql_error).
    """
    total = len(eval_results)
    if total == 0:
        return {"success_rate": 0.0, "successful": 0, "total": 0}
    
    successful = 0
    for result in eval_results:
        # Check if SQL was generated (empty SQL is a failure)
        generated_sql = result.get("generated_sql")
        has_sql = generated_sql is not None and str(generated_sql).strip() != ""
        
        # Support multiple formats:
        # - New format: sql_execution_ok
        # - Spatial-RAG format: sql_executed_ok
        # - Old format: sql_failed (inverted)
        if "sql_execution_ok" in result:
            sql_ok = result.get("sql_execution_ok", False)
        elif "sql_executed_ok" in result:
            sql_ok = result.get("sql_executed_ok", False)
        else:
            sql_ok = not result.get("sql_failed", False)
        
        routing_ok = not result.get("routing_failed", False)
        
        # Check for errors in both old and new formats
        error = result.get("error")
        sql_error = result.get("sql_error")
        no_error = error is None and sql_error is None
        
        # Success = SQL was generated AND pipeline ran without errors (regardless of POI count)
        if has_sql and sql_ok and routing_ok and no_error:
            successful += 1
    
    success_rate = successful / total if total > 0 else 0.0
    
    return {
        "success_rate": success_rate,
        "successful": successful,
        "total": total,
        "failed": total - successful,
    }


def compute_avg_conversation_turns(eval_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute average conversation turns."""
    if not eval_results:
        return {"avg_turns": 0.0, "min_turns": 0, "max_turns": 0, "total": 0}
    
    turns = [result.get("num_turns", 0) for result in eval_results]
    
    return {
        "avg_turns": sum(turns) / len(turns) if turns else 0.0,
        "min_turns": min(turns) if turns else 0,
        "max_turns": max(turns) if turns else 0,
        "total": len(turns),
    }


def compute_ambiguity_detection_f1(eval_results: List[Dict[str, Any]], gt_metadata_lookup: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute F1 score for query-level ambiguity detection (binary classification).
    
    Supports both pred_is_ambiguous and pred_ambiguity field names.
    GT ambiguity is derived from ambiguity_types (non-empty = ambiguous) or gt_is_ambiguous.
    If gt_metadata_lookup is provided, uses it to get GT ambiguity info.
    """
    # Load GT metadata if not in results
    if gt_metadata_lookup is None:
        # Try to load from standard location
        gt_file = Path("/workspace/storage/DSPP-CH/Final-Data/Full-Data/EVClarify_queries.json")
        if gt_file.exists():
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                gt_metadata_lookup = {item["query_id"]: item for item in gt_data}
            except Exception:
                gt_metadata_lookup = {}
    
    tp = fp = fn = tn = 0
    
    for result in eval_results:
        # Predicted ambiguity: use identified_entropies (non-null/non-empty = ambiguous)
        identified_entropies = result.get("identified_entropies")
        if identified_entropies is not None and len(identified_entropies) > 0:
            pred_amb = True
        else:
            # Fallback to other field names
            pred_amb = result.get("pred_is_ambiguous", False)
            if not pred_amb:
                pred_amb = result.get("pred_ambiguity", False)
        
        # GT ambiguity: check multiple sources
        # ["Unambiguous"] or empty = unambiguous, otherwise ambiguous
        gt_amb = result.get("gt_is_ambiguous", False)
        if not gt_amb:
            # Check ambiguity_types in result
            ambiguity_types = result.get("ambiguity_types", [])
            if not ambiguity_types:
                # Try gt_ambiguity_types
                ambiguity_types = result.get("gt_ambiguity_types", [])
            if not ambiguity_types and gt_metadata_lookup:
                # Load from GT metadata lookup
                query_id = result.get("query_id")
                if query_id in gt_metadata_lookup:
                    ambiguity_types = gt_metadata_lookup[query_id].get("ambiguity_types", [])
            
            # Check if unambiguous: ["Unambiguous"] or empty list means unambiguous
            if not ambiguity_types or ambiguity_types == ["Unambiguous"]:
                gt_amb = False
            else:
                gt_amb = True
        
        if pred_amb and gt_amb:
            tp += 1
        elif pred_amb and not gt_amb:
            fp += 1
        elif not pred_amb and gt_amb:
            fn += 1
        else:  # not pred_amb and not gt_amb
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": tp + fp + fn + tn,
    }


def map_gt_ambiguity_to_slots(gt_ambiguity_types: List[str]) -> Set[str]:
    """Map GT ambiguity types to slot names."""
    slot_map = {
        "Toponymic": "Anchor",
        "Target": "Target",
        "Mode": "Mode",
        "Threshold": "Threshold",
    }
    
    slots = set()
    for amb_type in gt_ambiguity_types:
        if amb_type in slot_map:
            slots.add(slot_map[amb_type])
    
    return slots


def compute_parameter_ambiguity_f1(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute F1 score for parameter-level ambiguity detection (multi-label classification)."""
    all_slots = {"Anchor", "Target", "Mode", "Threshold"}
    
    # Per-slot metrics
    slot_metrics = {}
    
    for slot in all_slots:
        tp = fp = fn = 0
        
        for result in eval_results:
            pred_slots = set(result.get("found_ambiguity_slots", []))
            gt_amb_types = result.get("gt_ambiguity_types", [])
            gt_slots = map_gt_ambiguity_to_slots(gt_amb_types)
            
            pred_has_slot = slot in pred_slots
            gt_has_slot = slot in gt_slots
            
            if pred_has_slot and gt_has_slot:
                tp += 1
            elif pred_has_slot and not gt_has_slot:
                fp += 1
            elif not pred_has_slot and gt_has_slot:
                fn += 1
            # else: tn (both absent) - not needed for F1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        slot_metrics[slot] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    # Macro-average (average across slots)
    macro_precision = sum(m["precision"] for m in slot_metrics.values()) / len(slot_metrics)
    macro_recall = sum(m["recall"] for m in slot_metrics.values()) / len(slot_metrics)
    macro_f1 = sum(m["f1"] for m in slot_metrics.values()) / len(slot_metrics)
    
    # Micro-average (pool all TP/FP/FN across slots)
    total_tp = sum(m["tp"] for m in slot_metrics.values())
    total_fp = sum(m["fp"] for m in slot_metrics.values())
    total_fn = sum(m["fn"] for m in slot_metrics.values())
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    return {
        "per_slot": slot_metrics,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
    }


def compute_identified_ambiguity_accuracy(eval_results: List[Dict[str, Any]], gt_metadata_lookup: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute accuracy for identified ambiguities against true ambiguity types.
    
    Compares identified_entropies (from eval results) against gt_ambiguity_types (from true intent).
    A query is marked as "good" if all identified ambiguities match all true ambiguity types.
    
    If identified_entropies is null, query is marked as unambiguous.
    
    Returns:
        Dict with accuracy, exact_matches (good), total, and per-query breakdown.
    """
    # Load GT metadata if not provided
    if gt_metadata_lookup is None:
        gt_file = Path("/workspace/storage/DSPP-CH/Final-Data/Full-Data/EVClarify_queries.json")
        if gt_file.exists():
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                gt_metadata_lookup = {item["query_id"]: item for item in gt_data}
            except Exception:
                gt_metadata_lookup = {}
    
    total = 0
    exact_matches = 0  # All identified match all GT (good)
    per_query_results = []
    
    for result in eval_results:
        # Extract identified slots from identified_entropies
        # If identified_entropies is null, mark as unambiguous (empty slots)
        identified_entropies = result.get("identified_entropies")
        if identified_entropies is None:
            identified_slots = set()  # Null means unambiguous
        else:
            identified_slots = {ent.get("slot", "") for ent in identified_entropies if ent.get("slot")}
        
        # Get GT ambiguity types and map to slots
        gt_ambiguity_types = result.get("gt_ambiguity_types", [])
        if not gt_ambiguity_types and gt_metadata_lookup:
            # Load from GT metadata lookup
            query_id = result.get("query_id")
            if query_id in gt_metadata_lookup:
                gt_ambiguity_types = gt_metadata_lookup[query_id].get("ambiguity_types", [])
        
        # Filter out "Unambiguous" from GT types (it's a marker, not an ambiguity type)
        if gt_ambiguity_types == ["Unambiguous"]:
            gt_ambiguity_types = []
        
        gt_slots = map_gt_ambiguity_to_slots(gt_ambiguity_types)
        
        # Compare: both should have same slots
        total += 1
        match = (identified_slots == gt_slots)
        if match:
            exact_matches += 1
        
        per_query_results.append({
            "query_id": result.get("query_id"),
            "identified_slots": list(identified_slots),
            "gt_slots": list(gt_slots),
            "match": match,
        })
    
    accuracy = exact_matches / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "exact_matches": exact_matches,  # Good (all identified match all GT)
        "total": total,
        "mismatches": total - exact_matches,
        "per_query": per_query_results,
    }


def normalize_poi_name(name: str) -> str:
    """Normalize POI name for comparison (lowercase, strip whitespace)."""
    return name.lower().strip() if name else ""


def compute_final_answer_f1(eval_results: List[Dict[str, Any]], gt_poi_lookup: Dict[str, Dict[str, Any]] = None, k: int = 5) -> Dict[str, Any]:
    """Compute F1@k for final answer accuracy using top-k GT vs top-k predicted POIs.
    
    Args:
        eval_results: List of evaluation results
        gt_poi_lookup: Optional lookup dict from GT_Chicago_POI.json. If None, uses gt_poi_list from eval_results.
        k: Top-k POIs to evaluate (default: 5 for F1@5, use 1 for F1@1)
    
    Only computes metrics for queries that have both predicted and GT POIs.
    """
    total_tp = total_fp = total_fn = 0
    per_query_results = []
    skipped_no_pred = 0
    skipped_no_gt = 0
    skipped_both_empty = 0
    
    for result in eval_results:
        query_id = result.get("query_id")
        pred_pois = result.get("final_poi", [])[:k]  # Top-k predicted
        
        # Get GT POIs: prefer gt_poi_lookup, fallback to result["gt_poi_list"]
        if gt_poi_lookup and query_id in gt_poi_lookup:
            gt_record = gt_poi_lookup[query_id]
            gt_pois_raw = gt_record.get("gt_poi_abnormal", [])
            # Convert to same format as eval_results
            gt_pois = [
                {
                    "poi_name": poi.get("name", ""),
                    "poi_type": "unknown",  # Not in GT_Chicago_POI.json
                    "route_cost_s": poi.get("travel_time_s", 0.0),
                    "distance_m": poi.get("distance_m", 0.0),
                }
                for poi in gt_pois_raw
            ][:k]
        else:
            gt_pois = result.get("gt_poi_list", [])[:k]  # Fallback to eval_results
        
        # Skip queries without both pred and GT
        if not pred_pois and not gt_pois:
            skipped_both_empty += 1
            continue
        if not pred_pois:
            skipped_no_pred += 1
            continue
        if not gt_pois:
            skipped_no_gt += 1
            continue
        
        # Extract POI names (normalized)
        pred_names = {normalize_poi_name(poi.get("poi_name", "")) for poi in pred_pois if poi.get("poi_name")}
        gt_names = {normalize_poi_name(poi.get("poi_name", "")) for poi in gt_pois if poi.get("poi_name")}
        
        # Remove empty strings
        pred_names = {n for n in pred_names if n}
        gt_names = {n for n in gt_names if n}
        
        # Skip if both are empty after normalization
        if not pred_names and not gt_names:
            skipped_both_empty += 1
            continue
        
        # Compute TP, FP, FN for this query
        tp = len(pred_names & gt_names)  # Intersection
        fp = len(pred_names - gt_names)  # In pred but not in GT
        fn = len(gt_names - pred_names)  # In GT but not in pred
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        per_query_results.append({
            "query_id": query_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "pred_count": len(pred_names),
            "gt_count": len(gt_names),
        })
    
    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Per-query average
    if per_query_results:
        avg_tp = sum(r["tp"] for r in per_query_results) / len(per_query_results)
        avg_fp = sum(r["fp"] for r in per_query_results) / len(per_query_results)
        avg_fn = sum(r["fn"] for r in per_query_results) / len(per_query_results)
    else:
        avg_tp = avg_fp = avg_fn = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "avg_tp_per_query": avg_tp,
        "avg_fp_per_query": avg_fp,
        "avg_fn_per_query": avg_fn,
        "total_queries": len(per_query_results),
        "skipped_no_pred": skipped_no_pred,
        "skipped_no_gt": skipped_no_gt,
        "skipped_both_empty": skipped_both_empty,
        "total_evaluated": len(per_query_results),
    }


def compute_final_answer_ndcg_at_k(
    eval_results: List[Dict[str, Any]],
    gt_poi_lookup: Dict[str, Dict[str, Any]] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """Compute nDCG@k for final POI ranking.

    Treats GT POIs as the relevant set (binary relevance) and evaluates the
    ranked list of predicted POIs up to position k.
    """

    def _get_pois_for_result(result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        query_id = result.get("query_id")
        pred_pois = result.get("final_poi", [])[:k]

        if gt_poi_lookup and query_id in gt_poi_lookup:
            gt_record = gt_poi_lookup[query_id]
            gt_pois_raw = gt_record.get("gt_poi_abnormal", [])
            gt_pois_local = [
                {
                    "poi_name": poi.get("name", ""),
                    "poi_type": "unknown",
                    "route_cost_s": poi.get("travel_time_s", 0.0),
                    "distance_m": poi.get("distance_m", 0.0),
                }
                for poi in gt_pois_raw
            ][:k]
        else:
            gt_pois_local = result.get("gt_poi_list", [])[:k]

        return pred_pois, gt_pois_local

    total_ndcg = 0.0
    total_dcg = 0.0
    total_idcg = 0.0
    evaluated = 0
    skipped_no_pred = 0
    skipped_no_gt = 0
    skipped_both_empty = 0

    for result in eval_results:
        pred_pois, gt_pois = _get_pois_for_result(result)

        # Skip if no predictions and no GT
        if not pred_pois and not gt_pois:
            skipped_both_empty += 1
            continue
        if not pred_pois:
            skipped_no_pred += 1
            continue
        if not gt_pois:
            skipped_no_gt += 1
            continue

        pred_names = [normalize_poi_name(p.get("poi_name", "")) for p in pred_pois]
        gt_names_set = {
            normalize_poi_name(p.get("poi_name", ""))
            for p in gt_pois
            if p.get("poi_name")
        }

        # Remove empty names from GT
        gt_names_set = {n for n in gt_names_set if n}

        # If GT has no valid names, skip
        if not gt_names_set:
            skipped_no_gt += 1
            continue

        # DCG@k
        dcg = 0.0
        for rank, name in enumerate(pred_names[:k], start=1):
            if not name:
                continue
            rel = 1.0 if name in gt_names_set else 0.0
            if rel > 0:
                dcg += rel / math.log2(rank + 1)

        # IDCG@k (ideal ranking: all relevant items first)
        max_rel = min(k, len(gt_names_set))
        idcg = 0.0
        for rank in range(1, max_rel + 1):
            idcg += 1.0 / math.log2(rank + 1)

        if idcg == 0.0:
            skipped_no_gt += 1
            continue

        ndcg = dcg / idcg
        total_ndcg += ndcg
        total_dcg += dcg
        total_idcg += idcg
        evaluated += 1

    avg_ndcg = total_ndcg / evaluated if evaluated > 0 else 0.0
    avg_dcg = total_dcg / evaluated if evaluated > 0 else 0.0
    avg_idcg = total_idcg / evaluated if evaluated > 0 else 0.0

    return {
        "k": k,
        "ndcg_at_k": avg_ndcg,
        "avg_dcg": avg_dcg,
        "avg_idcg": avg_idcg,
        "total_evaluated": evaluated,
        "skipped_no_pred": skipped_no_pred,
        "skipped_no_gt": skipped_no_gt,
        "skipped_both_empty": skipped_both_empty,
    }


def load_gt_sql_and_queries(base_path: Path = None) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load GT SQL and queries with true_intent.
    
    Args:
        base_path: Base path. Defaults to /workspace/storage/DSPP-CH.
    
    Returns:
        Tuple of (gt_sql_lookup, queries_lookup) both keyed by query_id
    """
    if base_path is None:
        base_path = DEFAULT_BASE
    
    # Load GT SQL
    gt_sql_file = base_path / "Final-Data" / "Full-Data" / "EVClarify_gt_sql.json"
    gt_sql_lookup = {}
    if gt_sql_file.exists():
        with open(gt_sql_file, 'r') as f:
            gt_sql_data = json.load(f)
        for item in gt_sql_data:
            query_id = item.get("query_id")
            if query_id:
                gt_sql_lookup[query_id] = item
    else:
        print(f"Warning: GT SQL file not found at {gt_sql_file}")
    
    # Load queries with true_intent
    queries_file = base_path / "Final-Data" / "Full-Data" / "EVClarify_queries.json"
    queries_lookup = {}
    if queries_file.exists():
        with open(queries_file, 'r') as f:
            queries_data = json.load(f)
        for item in queries_data:
            query_id = item.get("query_id")
            if query_id:
                queries_lookup[query_id] = item
    else:
        print(f"Warning: Queries file not found at {queries_file}")
    
    return gt_sql_lookup, queries_lookup


def normalize_intent_for_comparison(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize intent for comparison (round coordinates, normalize strings)."""
    normalized = {}
    
    # Anchor
    if "anchor" in intent:
        anchor = intent["anchor"]
        normalized["anchor"] = {
            "name": (anchor.get("name") or "").lower().strip(),
            "lat": round(float(anchor.get("lat", 0)), 6) if anchor.get("lat") is not None else None,
            "lon": round(float(anchor.get("lon", 0)), 6) if anchor.get("lon") is not None else None,
        }
    
    # Target
    normalized["target"] = (intent.get("target") or intent.get("target_type") or "").lower().strip()
    
    # Threshold
    if "threshold" in intent:
        threshold = intent["threshold"]
        threshold_m = threshold.get("threshold_m")
        if threshold_m is None and "distance_threshold_km" in intent:
            threshold_m = float(intent["distance_threshold_km"]) * 1000.0
        normalized["threshold_m"] = round(float(threshold_m), 1) if threshold_m is not None else None
    elif "distance_threshold_km" in intent:
        normalized["threshold_m"] = round(float(intent["distance_threshold_km"]) * 1000.0, 1)
    else:
        normalized["threshold_m"] = None
    
    # Mode
    normalized["mode"] = (intent.get("mode") or intent.get("nav_mode") or "").lower().strip()
    
    return normalized


def _argmax_with_random_tie(probs: Dict[str, float]) -> Optional[str]:
    """Return key with max probability; break ties randomly."""
    if not probs:
        return None
    max_val = max(probs.values())
    candidates = [k for k, v in probs.items() if abs(v - max_val) < 1e-9]
    return random.choice(candidates) if len(candidates) > 1 else candidates[0]


def _get_city_abbr_from_query(query_record: Dict[str, Any]) -> str:
    """Get city abbreviation from query record."""
    city_name_to_abbr = {
        "Chicago": "CH",
        "Houston": "HO",
        "New York": "NY",
        "San Antonio": "SA",
        "New Orleans": "NO",
        "Miami": "MI",
    }
    city = query_record.get("city", "Chicago")
    return city_name_to_abbr.get(city, "CH")


def compute_parser_accuracy(eval_results: List[Dict[str, Any]], queries_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Parser Accuracy (PA): % of queries where initial parsed intent matches true intent.
    
    Compares initial_parsed_intent from eval_results against true_intent from queries.
    Calculates per-parameter accuracy (anchor, target, threshold, mode) and aggregate accuracy.
    
    Equations:
    - Per-slot accuracy: Accuracy_s = Correct_s / Total_s
    - Aggregate accuracy: (Accuracy_anchor_name + Accuracy_target + Accuracy_threshold + Accuracy_mode) / 4
    
    For anchor: queries DB with parsed anchor_location, takes first candidate, compares name only (not coordinates).
    For target/mode: uses argmax from probabilities with random tie-breaking. 
                     If multiple targets/modes have equal probabilities and GT has ambiguity, 
                     considers it correct if true value is among the tied candidates.
    For threshold: if null/unknown/0.0, checks GT threshold ambiguity - if ambiguous, null is correct.
    """
    # Per-slot counters
    anchor_total = anchor_correct = anchor_name_correct = 0
    target_total = target_correct = 0
    threshold_total = threshold_correct = 0
    mode_total = mode_correct = 0
    
    # Load test_queries.json to get city info
    base_path = DEFAULT_BASE
    test_queries_file = base_path / "Final-Data" / "test_queries.json"
    city_lookup = {}
    if test_queries_file.exists():
        with open(test_queries_file, 'r') as f:
            test_queries = json.load(f)
        for q in test_queries:
            query_id = q.get("query_id")
            if query_id:
                city_lookup[query_id] = q.get("city", "Chicago")
    
    for result in eval_results:
        query_id = result.get("query_id")
        if not query_id or query_id not in queries_lookup:
            continue
        
        true_intent = queries_lookup[query_id].get("true_intent")
        if not true_intent:
            continue
        
        gt_ambiguity_types = queries_lookup[query_id].get("ambiguity_types", [])
        has_threshold_ambiguity = "Threshold" in gt_ambiguity_types
        
        # Detect format: Spatial-RAG has semantic_intent, IG has initial_parsed_intent
        is_spatial_rag = "semantic_intent" in result or "spatial_info" in result
        initial_parsed = result.get("initial_parsed_intent")
        
        # Get city for DB query
        city = city_lookup.get(query_id, "Chicago")
        city_abbr = _get_city_abbr_from_query({"city": city})
        
        # 1. Anchor accuracy: query DB, take first candidate, compare name + coordinates
        # Count ALL queries for anchor (treat null/missing as wrong)
        anchor_total += 1
        
        if is_spatial_rag:
            # Spatial-RAG format: use anchor_name field
            anchor_name_raw = result.get("anchor_name")
            anchor_location = (anchor_name_raw or "").strip() if anchor_name_raw is not None else ""
        else:
            # IG format: use initial_parsed_intent
            if initial_parsed:
                anchor_location_raw = initial_parsed.get("anchor_location")
                anchor_location = (anchor_location_raw or "").strip() if anchor_location_raw is not None else ""
            else:
                anchor_location = ""  # Missing initial_parsed_intent = wrong
        
        if anchor_location and anchor_location.lower() != "unknown" and uq is not None:
            try:
                db_candidates = uq.anchor_candidates_from_db(city_abbr, anchor_location)
                if db_candidates:
                    # Take first candidate from DB
                    first_candidate = db_candidates[0]
                    db_name = (first_candidate[0] or "").lower().strip()
                    db_lat = float(first_candidate[1]) if first_candidate[1] is not None else None
                    db_lon = float(first_candidate[2]) if first_candidate[2] is not None else None
                    
                    true_anchor = true_intent.get("anchor", {})
                    true_name = (true_anchor.get("name") or "").lower().strip()
                    true_lat = round(float(true_anchor.get("lat", 0)), 6) if true_anchor.get("lat") is not None else None
                    true_lon = round(float(true_anchor.get("lon", 0)), 6) if true_anchor.get("lon") is not None else None
                    
                    # Compare name and coordinates
                    name_match = db_name == true_name
                    lat_match = (db_lat is not None and true_lat is not None and 
                                abs(round(db_lat, 6) - true_lat) < 0.0001)
                    lon_match = (db_lon is not None and true_lon is not None and 
                                abs(round(db_lon, 6) - true_lon) < 0.0001)
                    
                    # Track name match only
                    if name_match:
                        anchor_name_correct += 1
                    
                    # Track name + coordinates match
                    if name_match and lat_match and lon_match:
                        anchor_correct += 1
            except Exception:
                pass  # DB query failed or no candidates - already counted, treat as wrong
        # else: anchor_location is empty/unknown - already counted, treat as wrong
        
        # 2. Target accuracy
        # Count ALL queries for target (treat null/missing as wrong)
        target_total += 1
        if is_spatial_rag:
            # Spatial-RAG format: use semantic_intent.type (string)
            semantic_intent = result.get("semantic_intent") or {}
            type_raw = semantic_intent.get("type")
            predicted_target_str = (type_raw or "").strip().lower() if type_raw is not None else ""
            true_target = (true_intent.get("target_type") or "").lower().strip()
            
            if predicted_target_str:
                if predicted_target_str == true_target:
                    target_correct += 1
            # else: missing target - already counted, treat as wrong
        else:
            # IG format: use probability dict
            if initial_parsed:
                target_probs = initial_parsed.get("target_poi_type", {})
                if isinstance(target_probs, dict) and target_probs and uq is not None:
                    # Normalize probabilities
                    target_probs_normalized = uq.complete_and_normalize(target_probs, uq.TARGET_TYPES)
                    
                    true_target = (true_intent.get("target_type") or "").lower().strip()
                    has_target_ambiguity = "Target" in gt_ambiguity_types
                    
                    # Check if any category has probability 1.0
                    max_prob = max(target_probs_normalized.values()) if target_probs_normalized else 0.0
                    has_single_1_0 = abs(max_prob - 1.0) < 1e-9
                    
                    # If GT has target ambiguity and predicted has no single category with 1.0 prob, consider correct
                    if has_target_ambiguity and not has_single_1_0:
                        # GT is ambiguous and parser doesn't commit to single target (no 1.0 prob) - correct
                        target_correct += 1
                    else:
                        # Normal case: use argmax (random tie-breaking)
                        predicted_target = _argmax_with_random_tie(target_probs_normalized)
                        if predicted_target and predicted_target.lower() == true_target:
                            target_correct += 1
                # else: missing target_probs - already counted, treat as wrong
            # else: missing initial_parsed - already counted, treat as wrong
        
        # 3. Threshold accuracy: handle null/unknown/0.0 with ambiguity check
        # Count ALL queries for threshold (treat null/missing as wrong unless GT has ambiguity)
        threshold_total += 1
        if is_spatial_rag:
            # Spatial-RAG format: use spatial_info.distance_km (convert to meters)
            spatial_info = result.get("spatial_info") or {}
            distance_km = spatial_info.get("distance_km")
            if distance_km is not None:
                try:
                    parsed_threshold_m = float(distance_km) * 1000.0
                except (ValueError, TypeError):
                    parsed_threshold_m = None
            else:
                parsed_threshold_m = None
        else:
            # IG format: use travel_threshold_meters
            if initial_parsed:
                parsed_threshold_m = initial_parsed.get("travel_threshold_meters")
            else:
                parsed_threshold_m = None  # Missing initial_parsed_intent
        
        true_threshold_km = true_intent.get("distance_threshold_km")
        true_threshold_m = float(true_threshold_km) * 1000.0 if true_threshold_km is not None else None
        
        # Check if parsed threshold is null/unknown/0.0
        is_null_or_zero = (parsed_threshold_m is None or 
                          parsed_threshold_m == 0.0 or 
                          str(parsed_threshold_m).lower() in ["null", "unknown", "none"])
        
        if is_null_or_zero and has_threshold_ambiguity:
            # Null is correct when query has threshold ambiguity
            threshold_correct += 1
        elif parsed_threshold_m is not None and true_threshold_m is not None:
            # Compare values (within 0.1m tolerance)
            if abs(float(parsed_threshold_m) - true_threshold_m) < 0.1:
                threshold_correct += 1
        # else: missing or wrong threshold - already counted, treat as wrong
        
        # 4. Mode accuracy: if GT has mode ambiguity and no category has 1.0 prob, consider correct
        # Count ALL queries for mode (treat null/missing as wrong)
        mode_total += 1
        
        if not is_spatial_rag:
            # IG format only (Spatial-RAG doesn't have mode info)
            if initial_parsed:
                mode_probs = initial_parsed.get("navigation_mode", {})
                if isinstance(mode_probs, dict) and mode_probs and uq is not None:
                    # Normalize probabilities
                    mode_probs_normalized = uq.complete_and_normalize(mode_probs, uq.NAV_MODES)
                    
                    true_mode = (true_intent.get("nav_mode") or "").lower().strip()
                    has_mode_ambiguity = "Mode" in gt_ambiguity_types
                    
                    # Check if any category has probability 1.0
                    max_prob = max(mode_probs_normalized.values()) if mode_probs_normalized else 0.0
                    has_single_1_0 = abs(max_prob - 1.0) < 1e-9
                    
                    # If GT has mode ambiguity and predicted has no single category with 1.0 prob, consider correct
                    if has_mode_ambiguity and not has_single_1_0:
                        # GT is ambiguous and parser doesn't commit to single mode (no 1.0 prob) - correct
                        mode_correct += 1
                    else:
                        # Normal case: use argmax (random tie-breaking)
                        predicted_mode = _argmax_with_random_tie(mode_probs_normalized)
                        if predicted_mode and predicted_mode.lower() == true_mode:
                            mode_correct += 1
                # else: missing mode_probs - already counted, treat as wrong
            # else: missing initial_parsed - already counted, treat as wrong
        # else: Spatial-RAG doesn't have mode - already counted, treat as wrong
    
    # Calculate per-slot accuracies
    anchor_accuracy = anchor_correct / anchor_total if anchor_total > 0 else 0.0
    anchor_name_accuracy = anchor_name_correct / anchor_total if anchor_total > 0 else 0.0
    target_accuracy = target_correct / target_total if target_total > 0 else 0.0
    threshold_accuracy = threshold_correct / threshold_total if threshold_total > 0 else 0.0
    mode_accuracy = mode_correct / mode_total if mode_total > 0 else 0.0
    
    # Aggregate accuracy: average of slot accuracies (using name-only for anchor)
    # If mode_total is 0 (e.g., Spatial-RAG), average only anchor, target, threshold
    if mode_total > 0:
        aggregate_accuracy = (anchor_name_accuracy + target_accuracy + threshold_accuracy + mode_accuracy) / 4.0
    else:
        aggregate_accuracy = (anchor_name_accuracy + target_accuracy + threshold_accuracy) / 3.0
    
    return {
        "parser_accuracy": aggregate_accuracy,  # Aggregate accuracy (for backward compatibility)
        "aggregate_accuracy": aggregate_accuracy,
        "per_slot_accuracy": {
            "anchor": {
                "accuracy": anchor_accuracy,  # Name + coordinates match
                "name_accuracy": anchor_name_accuracy,  # Name match only
                "correct": anchor_correct,  # Name + coordinates match
                "name_correct": anchor_name_correct,  # Name match only
                "total": anchor_total,
            },
            "target": {
                "accuracy": target_accuracy,
                "correct": target_correct,
                "total": target_total,
            },
            "threshold": {
                "accuracy": threshold_accuracy,
                "correct": threshold_correct,
                "total": threshold_total,
            },
            "mode": {
                "accuracy": mode_accuracy,
                "correct": mode_correct,
                "total": mode_total,
            },
        },
        "total_queries": len([r for r in eval_results if r.get("query_id") in queries_lookup]),
    }


def compute_syntax_error_rate(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Syntax Error Rate (SER): % of generated SQL queries that failed due to syntax error.
    
    Supports both old format (sql_failed, error) and new format (sql_execution_ok, sql_error).
    """
    total = 0
    syntax_errors = 0
    
    for result in eval_results:
        generated_sql = result.get("generated_sql")
        if generated_sql is None:
            continue
        
        total += 1
        
        # Check if SQL execution failed (support multiple formats)
        sql_failed = False
        if "sql_execution_ok" in result:
            sql_failed = not result.get("sql_execution_ok", False)
        elif "sql_executed_ok" in result:
            sql_failed = not result.get("sql_executed_ok", False)
        else:
            sql_failed = result.get("sql_failed", False)
        
        # Get error message (check both old and new formats)
        error = result.get("error", "")
        sql_error = result.get("sql_error", "")
        error_text = str(error) + " " + str(sql_error)
        
        # Check if error is syntax-related
        if sql_failed or error or sql_error:
            error_lower = error_text.lower()
            syntax_keywords = ["syntax", "parse", "invalid", "malformed", "sql generation failed", "parser error"]
            if any(keyword in error_lower for keyword in syntax_keywords):
                syntax_errors += 1
    
    error_rate = syntax_errors / total if total > 0 else 0.0
    
    return {
        "syntax_error_rate": error_rate,
        "syntax_errors": syntax_errors,
        "total": total,
        "valid": total - syntax_errors,
    }


def normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize candidate POI for comparison."""
    lat_val = candidate.get("lat")
    lon_val = candidate.get("lon") or candidate.get("long")
    
    return {
        "name": (candidate.get("name") or "").lower().strip(),
        "lat": round(float(lat_val), 6) if lat_val is not None else None,
        "lon": round(float(lon_val), 6) if lon_val is not None else None,
    }


def compute_execution_accuracy(eval_results: List[Dict[str, Any]], gt_sql_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Execution Accuracy (EX): % of generated SQL queries with same output as GT after execution.
    
    Compares generated_sql_candidate_list (from candidates file, loaded by load_eval_results) against gt_candidates.
    Does NOT execute SQL - uses candidates from eval results only.
    
    Counts ALL queries with GT SQL. TSR failures with no candidates are counted as mismatches
    (unless GT is also empty, in which case it's a match - both correctly returned no results).
    """
    total = 0
    exact_matches = 0
    
    for result in eval_results:
        query_id = result.get("query_id")
        if not query_id or query_id not in gt_sql_lookup:
            continue
        
        # Count ALL queries with GT SQL (including TSR failures)
        # TSR failures with no candidates are mismatches (unless GT is also empty)
        
        # Get candidates from eval results (already loaded from candidates file by load_eval_results)
        # Use generated_sql_candidate_list (from candidates file), NOT final_candidates
        pred_candidates = result.get("generated_sql_candidate_list", [])
        if not pred_candidates:
            pred_candidates = result.get("candidate_list", [])
        
        # Skip if candidates are strings (names only) - they should be full objects
        # This should not happen if candidates were properly loaded/updated
        if pred_candidates and isinstance(pred_candidates[0], str):
            # Candidates are strings, not full objects - skip this query
            # (This should be fixed by running update_spatial_rag_candidates.py)
            continue
        
        gt_candidates = gt_sql_lookup[query_id].get("gt_candidates", [])
        
        # If both are empty, consider it a match
        if not pred_candidates and not gt_candidates:
            total += 1
            exact_matches += 1
            continue
        
        # If only one is empty, it's a mismatch (count it)
        # - No predicted but GT has candidates: wrong (mismatch)
        # - Predicted has candidates but GT is empty: wrong (mismatch)
        if not pred_candidates or not gt_candidates:
            total += 1
            continue  # Count as mismatch (exact_matches not incremented)
        
        total += 1
        
        # Normalize and compare candidate sets
        pred_set = {tuple(sorted(normalize_candidate(c).items())) for c in pred_candidates}
        gt_set = {tuple(sorted(normalize_candidate(c).items())) for c in gt_candidates}
        
        if pred_set == gt_set:
            exact_matches += 1
    
    accuracy = exact_matches / total if total > 0 else 0.0
    
    return {
        "execution_accuracy": accuracy,
        "exact_matches": exact_matches,
        "total": total,
        "mismatches": total - exact_matches,
    }


def compute_jaccard_index(eval_results: List[Dict[str, Any]], gt_sql_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Jaccard Index (JAC): intersection-over-union similarity between predicted and GT SQL result sets.
    
    Uses generated_sql_candidate_list (from candidates file, loaded by load_eval_results).
    Does NOT execute SQL - uses candidates from eval results only.
    Uses the same query set as EX (all queries with GT SQL, including TSR failures).
    """
    total = 0
    total_jaccard = 0.0
    
    for result in eval_results:
        query_id = result.get("query_id")
        if not query_id or query_id not in gt_sql_lookup:
            continue
        
        # Count ALL queries with GT SQL (including TSR failures, same as EX)
        # TSR failures with no candidates get Jaccard = 0.0 (unless GT is also empty)
        
        # Get candidates from eval results (already loaded from candidates file by load_eval_results)
        # Use generated_sql_candidate_list (from candidates file), NOT final_candidates
        pred_candidates = result.get("generated_sql_candidate_list", [])
        if not pred_candidates:
            pred_candidates = result.get("candidate_list", [])
        
        # Skip if candidates are strings (names only) - they should be full objects
        # This should not happen if candidates were properly loaded/updated
        if pred_candidates and isinstance(pred_candidates[0], str):
            # Candidates are strings, not full objects - skip this query
            # (This should be fixed by running update_spatial_rag_candidates.py)
            continue
        
        gt_candidates = gt_sql_lookup[query_id].get("gt_candidates", [])
        
        total += 1  # Count all queries with GT SQL (same as EX)
        
        # If both are empty, consider it a perfect match (Jaccard = 1.0)
        if not pred_candidates and not gt_candidates:
            total_jaccard += 1.0
            continue
        
        # If only one is empty, Jaccard = 0.0 (count it, same as EX)
        if not pred_candidates or not gt_candidates:
            total_jaccard += 0.0
            continue
        
        # Normalize and create sets
        pred_set = {tuple(sorted(normalize_candidate(c).items())) for c in pred_candidates}
        gt_set = {tuple(sorted(normalize_candidate(c).items())) for c in gt_candidates}
        
        # Jaccard = |intersection| / |union|
        intersection = pred_set & gt_set
        union = pred_set | gt_set
        
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        total_jaccard += jaccard
    
    avg_jaccard = total_jaccard / total if total > 0 else 0.0
    
    return {
        "jaccard_index": avg_jaccard,
        "total": total,
        "total_jaccard_sum": total_jaccard,
    }


def compute_all_metrics(eval_json_path: Path, gt_json_path: Path = None, gt_poi_path: Path = None, auto_load_city_gt: bool = True) -> Dict[str, Any]:
    """Compute all metrics and return a comprehensive report.
    
    Args:
        eval_json_path: Path to evaluation results JSON
        gt_json_path: Optional path to GT metadata JSON. If None, extracts from eval_results.
        gt_poi_path: Optional path to GT POI JSON (GT_Chicago_POI.json). If None and auto_load_city_gt=True,
                     automatically loads city-wise GT files.
        auto_load_city_gt: If True and gt_poi_path is None, auto-load city GT files.
    """
    eval_results, gt_metadata_lookup, gt_poi_lookup = load_eval_results(eval_json_path, gt_json_path, gt_poi_path, auto_load_city_gt)
    
    # Load GT SQL and queries for new metrics
    gt_sql_lookup, queries_lookup = load_gt_sql_and_queries()
    
    metrics = {
        "task_success_rate": compute_task_success_rate(eval_results),
        "avg_conversation_turns": compute_avg_conversation_turns(eval_results),
        "ambiguity_detection_f1": compute_ambiguity_detection_f1(eval_results),
        "identified_ambiguity_accuracy": compute_identified_ambiguity_accuracy(eval_results, gt_metadata_lookup),
        "parser_accuracy": compute_parser_accuracy(eval_results, queries_lookup),
        "syntax_error_rate": compute_syntax_error_rate(eval_results),
        "execution_accuracy": compute_execution_accuracy(eval_results, gt_sql_lookup),
        "jaccard_index": compute_jaccard_index(eval_results, gt_sql_lookup),
    }
    
    return metrics


def print_metrics_report(metrics: Dict[str, Any], is_baseline: bool = False) -> None:
    """Print a formatted metrics report.

    Args:
        metrics: Metrics dictionary from compute_all_metrics.
        is_baseline: If True, hides metrics that are not meaningful for the
            baseline run (no dialogue / ambiguity module), such as average
            conversation turns and query-level ambiguity F1.
    """
    print("=" * 80)
    print("EVALUATION METRICS REPORT")
    print("=" * 80)
    print()
    
    # Task Success Rate
    tsr = metrics["task_success_rate"]
    print("1. Task Success Rate")
    print(f"   Success Rate: {tsr['success_rate']:.2%} ({tsr['successful']}/{tsr['total']})")
    print(f"   Failed: {tsr['failed']}")
    print()
    
    if not is_baseline:
        # Average Conversation Turns
        turns = metrics["avg_conversation_turns"]
        print("2. Average Conversation Turns")
        print(f"   Average: {turns['avg_turns']:.2f}")
        print(f"   Min: {turns['min_turns']}, Max: {turns['max_turns']}")
        print()
        
        # Query Ambiguity Detection F1
        amb_f1 = metrics["ambiguity_detection_f1"]
        print("3. Query Ambiguity Detection (F1)")
        print(f"   Precision: {amb_f1['precision']:.4f}")
        print(f"   Recall:    {amb_f1['recall']:.4f}")
        print(f"   F1:        {amb_f1['f1']:.4f}")
        print(f"   Accuracy:  {amb_f1['accuracy']:.4f}")
        print(f"   TP: {amb_f1['tp']}, FP: {amb_f1['fp']}, FN: {amb_f1['fn']}, TN: {amb_f1['tn']}")
        print()
        
        # Identified Ambiguity Accuracy (comparing identified_entropies vs gt_ambiguity_types)
        id_amb_acc = metrics["identified_ambiguity_accuracy"]
        print("4. Identified Ambiguity Accuracy")
        print(f"   Accuracy: {id_amb_acc['accuracy']:.2%} ({id_amb_acc['exact_matches']}/{id_amb_acc['total']})")
        print(f"   Exact Matches (Good): {id_amb_acc['exact_matches']}")
        print(f"   Mismatches: {id_amb_acc['mismatches']}")
        print()
    else:
        # For baseline runs, we skip dialogue / ambiguity metrics that are not
        # meaningful (no clarification turns, no ambiguity detection module).
        print("2–3. Dialogue / Ambiguity metrics skipped for baseline run (no turns or ambiguity module).")
        print()
    
    # Parser Accuracy (PA)
    pa = metrics["parser_accuracy"]
    print("5. Parser Accuracy (PA)" if not is_baseline else "4. Parser Accuracy (PA)")
    print(f"   Aggregate Accuracy: {pa.get('aggregate_accuracy', pa.get('parser_accuracy', 0.0)):.2%}")
    print()
    
    if "per_slot_accuracy" in pa:
        print("   Per-Slot Accuracy:")
        for slot_name, slot_metrics in pa["per_slot_accuracy"].items():
            acc = slot_metrics.get("accuracy", 0.0)
            correct = slot_metrics.get("correct", 0)
            total = slot_metrics.get("total", 0)
            
            # Special handling for anchor: show both name and name+coords
            if slot_name == "anchor" and "name_accuracy" in slot_metrics:
                name_acc = slot_metrics.get("name_accuracy", 0.0)
                name_correct = slot_metrics.get("name_correct", 0)
                print(f"     {slot_name.capitalize()}:")
                print(f"       Name match only: {name_acc:.2%} ({name_correct}/{total})")
                print(f"       Name + coordinates: {acc:.2%} ({correct}/{total})")
            else:
                print(f"     {slot_name.capitalize()}: {acc:.2%} ({correct}/{total})")
    else:
        # Fallback for old format
        print(f"   Accuracy: {pa.get('parser_accuracy', 0.0):.2%} ({pa.get('correct', 0)}/{pa.get('total', 0)})")
        print(f"   Correct: {pa.get('correct', 0)}, Incorrect: {pa.get('incorrect', 0)}")
    print()
    
    # Syntax Error Rate (SER)
    ser = metrics["syntax_error_rate"]
    print("6. Syntax Error Rate (SER)" if not is_baseline else "5. Syntax Error Rate (SER)")
    print(f"   Error Rate: {ser['syntax_error_rate']:.2%} ({ser['syntax_errors']}/{ser['total']})")
    print(f"   Syntax Errors: {ser['syntax_errors']}, Valid: {ser['valid']}")
    print()
    
    # Execution Accuracy (EX)
    ex = metrics["execution_accuracy"]
    print("7. Execution Accuracy (EX)" if not is_baseline else "6. Execution Accuracy (EX)")
    print(f"   Accuracy: {ex['execution_accuracy']:.2%} ({ex['exact_matches']}/{ex['total']})")
    print(f"   Exact Matches: {ex['exact_matches']}, Mismatches: {ex['mismatches']}")
    print()
    
    # Jaccard Index (JAC)
    jac = metrics["jaccard_index"]
    print("8. Jaccard Index (JAC)" if not is_baseline else "7. Jaccard Index (JAC)")
    print(f"   Average Jaccard: {jac['jaccard_index']:.4f}")
    print(f"   Queries evaluated: {jac['total']}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--eval_json", type=str, help="Path to evaluation results JSON")
    parser.add_argument("--gt_json", type=str, help="Optional path to GT metadata JSON. If not provided, extracts GT metadata from eval_results.")
    parser.add_argument("--gt_poi_json", type=str, help="Path to GT POI JSON (GT_Chicago_POI.json). If not provided, uses gt_poi_list from eval_results.")
    parser.add_argument("--output", type=str, help="Path to save metrics JSON (optional)")
    parser.add_argument(
        "--set",
        type=str,
        choices=["test", "baseline"],
        help="Shortcut to select default eval result set: 'test' (IG eval) or 'baseline' (DIRECT_PROMPT baseline).",
    )
    
    args = parser.parse_args()
    
    # Resolve evaluation JSON path: explicit path wins; otherwise use --set shortcut.
    if args.eval_json:
        eval_path = Path(args.eval_json)
    else:
        if not args.set:
            raise SystemExit("Either --eval_json must be provided or --set must be one of ['test', 'baseline'].")
        repo_root = Path(__file__).resolve().parents[2]
        if args.set == "test":
            eval_path = repo_root / "Result" / "test" / "eval_gpt_4o_mini_ALL.json"
        elif args.set == "baseline":
            eval_path = repo_root / "Result" / "baseline" / "baseline_gpt_4o_mini_ALL.json"
        else:
            raise SystemExit(f"Unrecognized set: {args.set}")

    gt_path = Path(args.gt_json) if args.gt_json else None
    gt_poi_path = Path(args.gt_poi_json) if args.gt_poi_json else None
    
    if not eval_path.exists():
        raise SystemExit(f"Evaluation JSON not found: {eval_path}")
    if gt_path and not gt_path.exists():
        raise SystemExit(f"GT JSON not found: {gt_path}")
    if gt_poi_path and not gt_poi_path.exists():
        raise SystemExit(f"GT POI JSON not found: {gt_poi_path}")
    
    metrics = compute_all_metrics(eval_path, gt_path, gt_poi_path, auto_load_city_gt=True)
    
    # Auto-detect if it's a baseline run (Spatial-RAG or other baseline formats)
    is_baseline = args.set == "baseline"
    if not is_baseline:
        # Check if results have Spatial-RAG format (no ambiguity module)
        try:
            with open(eval_path, 'r') as f:
                sample_results = json.load(f)
            if sample_results and isinstance(sample_results, list) and len(sample_results) > 0:
                sample = sample_results[0]
                # Spatial-RAG has semantic_intent/spatial_info but no identified_entropies
                has_spatial_rag_format = ("semantic_intent" in sample or "spatial_info" in sample)
                has_no_ambiguity_module = (sample.get("identified_entropies") is None and 
                                          sample.get("identified_slots") is None)
                if has_spatial_rag_format and has_no_ambiguity_module:
                    is_baseline = True
        except Exception:
            pass  # If detection fails, use default
    
    print_metrics_report(metrics, is_baseline=is_baseline)
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {output_path}")

