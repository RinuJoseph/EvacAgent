"""
Uncertainty quantification utilities for the method code.

"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIError, RateLimitError

import prompts

# Import paths from paths.py for consistency
import paths
BASE = paths.BASE
EMB_MODEL_PATH = BASE.parent / "EMB-Models/all-MiniLM-L6-v2"

TARGET_TYPES = [
    "hospital",
    "clinic",
    "doctors",
    "pharmacy",
    "fire_station",
    "police",
    "shelter",
    "community_centre",
    "school",
]
NAV_MODES = ["drive", "walk"]


import os
DEFAULT_TARGET_MIN_PROB = float(os.environ.get("DEFAULT_TARGET_MIN_PROB", "0.20"))
DEFAULT_MODE_MIN_PROB = float(os.environ.get("DEFAULT_MODE_MIN_PROB", "0.60"))


def load_cp_min_probs(city_abbr: str) -> Tuple[float, float, Optional[str]]:
    """
    Load CP-derived min probability cutoffs from:
      Final-Data/calib/thresholds_target.json
      Final-Data/calib/thresholds_mode.json
    
    Falls back to Data/GT_<CITY>/calib/ if Final-Data doesn't exist (legacy support).

    Returns:
      (min_prob_target, min_prob_mode, source_path_or_None)
    """
    abbr = str(city_abbr).strip().upper()
    # Map city abbreviations to full city names
    city_map = {
        "CH": "Chicago",
        "CHICAGO": "Chicago",
    }
    city_name = city_map.get(abbr, "Chicago")  
    
  
    target_path = BASE / "Final-Data" / "calib" / "thresholds_target.json"
    mode_path = BASE / "Final-Data" / "calib" / "thresholds_mode.json"
    
   
    if not target_path.exists():
        target_path = BASE / "Data" / f"GT_{city_name}" / "calib" / "thresholds_target.json"
    if not mode_path.exists():
        mode_path = BASE / "Data" / f"GT_{city_name}" / "calib" / "thresholds_mode.json"
    
    min_t = DEFAULT_TARGET_MIN_PROB
    min_m = DEFAULT_MODE_MIN_PROB
    source_paths = []
    
    # Load target threshold
    if target_path.exists():
        try:
            target_obj = json.loads(target_path.read_text(encoding="utf-8"))
            min_t = float(target_obj["target_type"]["min_prob"])
            min_t = max(0.0, min(1.0, min_t))  # safety: clamp into [0,1]
            source_paths.append(str(target_path))
        except Exception:
            pass
    
    # Load mode threshold
    if mode_path.exists():
        try:
            mode_obj = json.loads(mode_path.read_text(encoding="utf-8"))
            min_m = float(mode_obj["nav_mode"]["min_prob"])
            min_m = max(0.0, min(1.0, min_m))  # safety: clamp into [0,1]
            source_paths.append(str(mode_path))
        except Exception:
            pass
    
    source_path = ", ".join(source_paths) if source_paths else None
    return min_t, min_m, source_path


def sample_threshold_distribution_with_raw(
    client: OpenAI,
    query: str,
    n: int = 20,
    temperature: float = 0.9,
) -> Dict[str, Any]:
    
    raw: List[str] = []
    parsed: List[Optional[float]] = []
    errors: List[str] = []
    print(f"  Starting threshold sampling: {n} samples...", flush=True)
    start_time = time.time()
    for i in range(int(n)):
        try:
            sample_start = time.time()
            print(f"  [Sample {i+1}/{n}] Calling LLM...", end=" ", flush=True)
            # Add small delay between samples to avoid rate limits (except for first sample)
            if i > 0:
                time.sleep(0.5)  # 500ms delay between samples
            txt = call_llm_threshold_sample(client, query=query, temperature=temperature, timeout=20.0)
            elapsed = time.time() - sample_start
            print(f"✓ ({elapsed:.1f}s)", flush=True)
            raw.append(txt)
            s = (txt or "").strip().lower()
            if s == "null":
                parsed.append(None)
                print(f"  Threshold sample {i+1}/{n}: {txt!r} → null")
            else:
                val = parse_threshold_sample(txt)
                if val is None:
                    # ignore unparsable response
                    errors.append(f"unparsable: {txt!r}")
                    print(f"  Threshold sample {i+1}/{n}: {txt!r} → ERROR (unparsable)")
                else:
                    parsed.append(float(val))
                    print(f"  Threshold sample {i+1}/{n}: {txt!r} → {val:.0f}m")
        except Exception as e:
            errors.append(str(e))
            elapsed = time.time() - sample_start
            print(f"✗ ERROR ({elapsed:.1f}s) → {str(e)}", flush=True)
            # Continue to next sample even if this one failed
    total_elapsed = time.time() - start_time
    print(f"  Threshold sampling complete: {len(parsed)}/{n} successful in {total_elapsed:.1f}s", flush=True)
    if errors:
        print(f"  Errors encountered: {len(errors)}", flush=True)
    return {"n_requested": int(n), "n_ok": len(parsed), "raw": raw, "parsed": parsed, "errors": errors}

def complete_and_normalize(d: Dict[str, Any], cats: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in cats:
        try:
            out[c] = float(d.get(c, 0.0))
        except Exception:
            out[c] = 0.0
    s = sum(out.values())
    if s <= 1e-12:
        return {c: 1.0 / len(cats) for c in cats}
    return {c: out[c] / s for c in cats}


def entropy_bits_from_probs(probs: Dict[str, float]) -> float:
    vals = [float(v) for v in probs.values() if float(v) > 0]
    s = sum(vals)
    if s <= 0:
        return 0.0
    h = 0.0
    for v in vals:
        p = v / s
        h -= p * (math.log(p) / math.log(2.0))
    if h < 0 and abs(h) < 1e-9:
        h = 0.0
    return float(h)


def heuristic_set_from_probs(probs: Dict[str, float], min_prob: float) -> List[str]:
    best_v: float = float("-inf")
    out: List[str] = []
    best_keys: List[str] = []
    for k, v in probs.items():
        fv = float(v)
        if fv > best_v:
            best_v = fv
            best_keys = [k]
        elif abs(fv - best_v) < 1e-9:
            best_keys.append(k)
        if fv >= min_prob:
            out.append(k)
    if not out and best_keys:
        out = best_keys
    out.sort()
    return out


# -----------------------------
# Reverse geocoding (best-effort)
# -----------------------------


def reverse_geocode_osm(lat: float, lon: float, timeout_s: float = 8.0) -> Optional[str]:
    try:
        import requests  
    except Exception:
        return None
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": str(lat), "lon": str(lon), "zoom": "18", "addressdetails": "1"}
    headers = {"User-Agent": "DSPP-CH/uq.py (research; contact: local)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)
        if resp.status_code != 200:
            return None
        data = resp.json()
        disp = data.get("display_name")
        if isinstance(disp, str) and disp.strip():
            return disp.strip()
        return None
    except Exception:
        return None


def guess_address_from_name(name: str) -> Optional[str]:
    s = (name or "").strip()
    if " - " in s:
        tail = s.split(" - ", 1)[1].strip()
        if re.match(r"^\d+\s+.+,\s*.+", tail):
            return tail
    return None

# Threshold sampling + clustering
def call_llm_threshold_sample(
    client: OpenAI,
    query: str,
    temperature: float = 0.7,
    max_retries: int = 6,
    timeout: float = 20.0,
) -> str:
    last_err: Exception | None = None
    
    prompt = prompts.THRESHOLD_SAMPLING_PROMPT.replace("{query}", query)
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=20,
                    timeout=timeout,
                )
            except TypeError:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=20,
                )
            if isinstance(resp, dict):
                content = resp.get("choices", [{}])[0].get("message", {}).get("content") or ""
            else:
                content = resp.choices[0].message.content or ""
            return content.strip()
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            last_err = e
            # For connection errors, fail faster (fewer retries)
            if isinstance(e, APIConnectionError):
                # Connection errors: only 2 retries, shorter waits
                if attempt < 2:
                    wait_time = min(2.0**attempt, 5.0)
                    print(f"    [Retry {attempt+1}/2] Error: {type(e).__name__}, waiting {wait_time:.1f}s...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"    [Connection failed] Error: {type(e).__name__}: {e}", flush=True)
                    break  # Fail fast on connection errors
            elif attempt < max_retries - 1:
                # Longer wait for rate limits
                if isinstance(e, RateLimitError):
                    wait_time = min(2.0**attempt * 2, 30.0)  # Up to 30s for rate limits
                else:
                    wait_time = min(2.0**attempt, 10.0)
                print(f"    [Retry {attempt+1}/{max_retries}] Error: {type(e).__name__}, waiting {wait_time:.1f}s...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"    [Final attempt failed] Error: {type(e).__name__}: {e}", flush=True)
        except Exception as e:
            last_err = e
            print(f"    [Unexpected error] {type(e).__name__}: {e}", flush=True)
            break
    # For connection errors, return a fallback value instead of raising
    if isinstance(last_err, APIConnectionError):
        print(f"    [Connection error - using fallback] Returning 'null' as fallback", flush=True)
        return "null"  # Return null as fallback for connection errors
    raise RuntimeError(f"Threshold sampling LLM call failed after {max_retries} retries: {last_err}")


def parse_threshold_sample(text: str) -> Optional[float]:
    s = (text or "").strip().lower()
    if not s:
        return None
    if s == "null":
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def sample_threshold_distribution(client: OpenAI, query: str = "", n: int = 20, temperature: float = 0.9) -> List[Optional[float]]:
    """
    Returns a list of samples where:
      - None represents "null" (true nearest, no max distance)
      - float represents meters
    """
    out: List[Optional[float]] = []
    for _ in range(n):
        txt = call_llm_threshold_sample(client, query=query, temperature=temperature)
        s = (txt or "").strip().lower()
        if s == "null":
            out.append(None)
            continue
        val = parse_threshold_sample(txt)
        if val is not None:
            out.append(float(val))
    return out


def cluster_1d_threshold_samples(samples_m: List[Optional[float]], eps_m: float = 2000.0) -> Dict[str, Any]:
    """
    Cluster threshold samples in 1D using DBSCAN:
      - numeric meters are clustered via DBSCAN
      - null is treated as its own cluster
    Compute entropy over cluster masses (bits).
    """
    if not samples_m:
        return {"n": 0, "eps_m": eps_m, "clusters": [], "entropy_bits": 0.0, "dominant": None}

    null_count = sum(1 for x in samples_m if x is None)
    numeric = [float(x) for x in samples_m if x is not None]

    clusters: List[Dict[str, Any]] = []

    n_total = len(samples_m)
    if null_count > 0:
        clusters.append(
            {
                "cluster": "null",
                "kind": "null",
                "count": int(null_count),
                "mass": float(null_count) / float(n_total),
                "center_m": None,
                "min_m": None,
                "max_m": None,
            }
        )

    if not numeric:
        H_bits = 0.0
        for c in clusters:
            p = float(c["mass"])
            if p > 0:
                H_bits -= p * (math.log(p) / math.log(2.0))
        if H_bits < 0 and abs(H_bits) < 1e-9:
            H_bits = 0.0
        dominant = max(clusters, key=lambda x: float(x["mass"])) if clusters else None
        return {"n": n_total, "eps_m": eps_m, "clusters": clusters, "entropy_bits": float(H_bits), "dominant": dominant}

    try:
        import numpy as np  # type: ignore
        from sklearn.cluster import DBSCAN  # type: ignore

        X = np.asarray(numeric, dtype=float).reshape(-1, 1)
        db = DBSCAN(eps=eps_m, min_samples=1, metric="euclidean").fit(X)
        raw = [int(x) for x in db.labels_.tolist()]
    except Exception:
        # fallback: treat all as one cluster
        raw = [0 for _ in numeric]

    uniq = sorted(set(raw))
    remap = {lb: i for i, lb in enumerate(uniq)}
    labels = [remap[lb] for lb in raw]

    counts = Counter(labels)
    n_num = len(numeric)
    for k in sorted(counts):
        members = [numeric[i] for i, lb in enumerate(labels) if lb == k]
        # Use a robust representative radius for the cluster:
        # - median of the member values, not a simple average over all samples.
        members_sorted = sorted(members)
        m_len = len(members_sorted)
        if m_len == 0:
            center_val = 0.0
        elif m_len % 2 == 1:
            center_val = members_sorted[m_len // 2]
        else:
            center_val = 0.5 * (members_sorted[m_len // 2 - 1] + members_sorted[m_len // 2])

        clusters.append(
            {
                "cluster": k,
                "kind": "meters",
                "count": counts[k],
                "mass": counts[k] / n_total,
                "center_m": float(center_val),
                "min_m": float(min(members)),
                "max_m": float(max(members)),
            }
        )

    H_bits = 0.0
    for c in clusters:
        p = float(c["mass"])
        if p > 0:
            H_bits -= p * (math.log(p) / math.log(2.0))
    if H_bits < 0 and abs(H_bits) < 1e-9:
        H_bits = 0.0

    dominant = max(clusters, key=lambda x: float(x["mass"])) if clusters else None
    return {"n": n_total, "eps_m": eps_m, "clusters": clusters, "entropy_bits": float(H_bits), "dominant": dominant}


# -----------------------------
# Anchor ambiguity (clustering + entropy)
# -----------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


_ST_MODEL: Any = None


def embed_texts_minilm(texts: List[str]) -> Optional[Any]:
    global _ST_MODEL
    try:
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    if not EMB_MODEL_PATH.exists():
        return None
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(str(EMB_MODEL_PATH))
    try:
        emb = _ST_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb)
    except Exception:
        return None


def dbscan_labels_from_precomputed(D: List[List[float]], eps: float, min_samples: int = 1) -> List[int]:
    try:
        import numpy as np  # type: ignore
        from sklearn.cluster import DBSCAN  # type: ignore

        D_np = np.asarray(D, dtype=float)
        cl = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(D_np)
        return [int(x) for x in cl.labels_.tolist()]
    except Exception:
        # very small connected-components fallback
        n = len(D)
        labels = [-1] * n
        cur = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            stack = [i]
            labels[i] = cur
            while stack:
                u = stack.pop()
                for v in range(n):
                    if labels[v] == -1 and D[u][v] <= eps:
                        labels[v] = cur
                        stack.append(v)
            cur += 1
        return labels


def normalize_labels(labels: List[int]) -> Tuple[List[int], int, List[int]]:
    uniq = sorted(set(labels))
    remap = {lb: i for i, lb in enumerate(uniq)}
    out = [remap[lb] for lb in labels]
    K = len(uniq)
    counts = Counter(out)
    sizes = [counts[k] for k in range(K)]
    return out, K, sizes


def entropy_bits_from_sizes(sizes: List[int]) -> float:
    n = sum(sizes)
    if n <= 0:
        return 0.0
    H = 0.0
    for c in sizes:
        p = c / n
        if p > 0:
            H -= p * (math.log(p) / math.log(2.0))
    if H < 0 and abs(H) < 1e-9:
        H = 0.0
    return float(H)


def build_semantic_text(addr: str) -> str:
    # Remove POI name and street chunk; keep neighborhood/district-ish chunks.
    parts = [p.strip() for p in str(addr).split(",") if p.strip()]
    if len(parts) >= 5:
        keep = parts[3:7]
    elif len(parts) >= 3:
        keep = parts[1:4]
    else:
        keep = parts
    return ", ".join(keep)


def anchor_candidates_from_db(city_abbr: str, anchor_text: str) -> List[Tuple[str, float, float, str]]:
    """
    Retrieve anchor candidates from database.
    
    Uses case-insensitive LIKE matching without normalization.
    """
    db_path = BASE / "DB" / "DSPP_DB.duckdb"
    if not db_path.exists():
        return []
    table = f"{city_abbr}_POI"
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        if table not in tables:
            return []
        
        # Simple case-insensitive LIKE pattern matching
        like_pattern = f"%{anchor_text.lower()}%"
        
        # No normalization - simple case-insensitive search
        query = f"""
        SELECT name, lat, long, fclass 
        FROM {table} 
        WHERE lower(name) LIKE ?
        """
        return con.execute(query, [like_pattern]).fetchall()
    finally:
        con.close()


def enrich_candidates(rows: List[Tuple[str, float, float, str]], shuffle_seed: int = 42, skip_reverse_geocode: bool = False) -> List[Dict[str, Any]]:
    rows2 = list(rows)
    random.Random(shuffle_seed).shuffle(rows2)

    cache: Dict[Tuple[int, int], Optional[str]] = {}
    last_call = 0.0
    out: List[Dict[str, Any]] = []
    for i, (name, lat, lon, fclass) in enumerate(rows2, start=1):
        addr_guess = guess_address_from_name(name)
        if skip_reverse_geocode:
            # Skip reverse geocoding for performance (not needed for clustering)
            addr = None
        else:
            key = (int(round(lat * 1e5)), int(round(lon * 1e5)))
            if key in cache:
                addr = cache[key]
            else:
                sleep_for = 1.0 - (time.time() - last_call)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                addr = reverse_geocode_osm(float(lat), float(lon))
                cache[key] = addr
                last_call = time.time()
        out.append(
            {
                "rank": i,
                "name": name,
                "fclass": fclass,
                "lat": float(lat),
                "lon": float(lon),
                "address": addr if addr else None,
                "address_guess": addr_guess if addr_guess else None,
            }
        )
    return out


def analyze_anchor_ambiguity(enriched: List[Dict[str, Any]], alpha: float = 0.5, eps: float = 0.28, semantic_eps: Optional[float] = None, eps_km: float = 5.0) -> Dict[str, Any]:
    """
    Analyze anchor ambiguity using spatial clustering only (Haversine distance).
    
    Args:
        enriched: List of enriched anchor candidates
        alpha: (deprecated, kept for compatibility)
        eps: (deprecated, kept for compatibility)
        semantic_eps: (deprecated, kept for compatibility)
        eps_km: Haversine distance threshold in kilometers (default: 5.0 km)
    
    Returns:
        Dictionary with clustering results and ambiguity flag
    """
    n = len(enriched)
    if n == 0:
        return {"n_candidates": 0, "anchor_ambiguous": False}

    # Spatial distance matrix using Haversine distance in km
    # Convert to normalized distance for DBSCAN: distance = 1 - (1 / (1 + haversine_km))
    D_spatial = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dh_km = haversine_km(enriched[i]["lat"], enriched[i]["lon"], enriched[j]["lat"], enriched[j]["lon"])
            d_spatial_sim = 1.0 / (1.0 + dh_km)  # similarity
            D_spatial[i][j] = 1.0 - d_spatial_sim

    # Convert eps_km to normalized distance space
    # For 5km: similarity = 1/(1+5) = 0.1667, distance = 1 - 0.1667 = 0.8333
    eps_normalized = 1.0 - (1.0 / (1.0 + eps_km))
    
    # Cluster using spatial distance only
    labs_sp_raw = dbscan_labels_from_precomputed(D_spatial, eps=eps_normalized, min_samples=1)
    labs_sp, K_sp, sizes_sp = normalize_labels(labs_sp_raw)
    H_sp = entropy_bits_from_sizes(sizes_sp)

    return {
        "n_candidates": n,
        "spatial_only": {"K": K_sp, "H_bits": H_sp, "sizes": sizes_sp, "labels": labs_sp, "eps_km": eps_km},
        "anchor_ambiguous": (H_sp > 1.0),  # Based on spatial entropy
    }


def run_uq(
    client: OpenAI,
    query: str,
    parsed: Dict[str, Any],
    city_abbr: str = "CH",
    target_min_prob: float = 0.20,
    mode_min_prob: float = 0.60,
) -> None:
    # categorical
    target_probs = complete_and_normalize(parsed.get("target_poi_type", {}) or {}, TARGET_TYPES)
    mode_probs = complete_and_normalize(parsed.get("navigation_mode", {}) or {}, NAV_MODES)
    target_set = heuristic_set_from_probs(target_probs, target_min_prob)
    mode_set = heuristic_set_from_probs(mode_probs, mode_min_prob)
    print("\nHeuristic set selection (fixed probability cutoffs; NOT conformal prediction):")
    print(f"- target_type: keep p >= {target_min_prob:.2f} → {target_set} (size={len(target_set)})")
    print(f"- nav_mode:    keep p >= {mode_min_prob:.2f} → {mode_set} (size={len(mode_set)})")
    print("Ambiguity analysis (categorical):")
    print(f"- target_type_ambiguous: {len(target_set) != 1} | entropy_bits={entropy_bits_from_probs(target_probs):.4f}")
    print(f"- nav_mode_ambiguous:    {len(mode_set) != 1} | entropy_bits={entropy_bits_from_probs(mode_probs):.4f}")

    # threshold sampling
    print("- threshold_sampling:")
    try:
        samples = sample_threshold_distribution(client, n=20, temperature=0.9)
        cl = cluster_1d_threshold_samples(samples, eps_m=2000.0)
        print(f"  - n={cl['n']} eps_m={cl['eps_m']} entropy_bits={cl['entropy_bits']:.4f}")
        if cl["dominant"] is not None:
            print(f"  - dominant={cl['dominant']}")
        print(f"  - clusters={cl['clusters']}")
        print(f"  - high_entropy(>1.2 bits): {float(cl['entropy_bits']) > 1.2}")
    except Exception as e:
        print(f"  - ERROR (skipping): {e}")

    # anchor
    anchor = str(parsed.get("anchor_location") or "").strip()
    print(f"\nAnchor parsed: {anchor or 'EMPTY'}")
    if anchor and anchor.lower() != "unknown":
        rows = anchor_candidates_from_db(city_abbr, anchor)
        print(f"DB matches for anchor '{anchor}': {len(rows)}")
        if rows:
            enriched = enrich_candidates(rows)
            print("\nShuffled DB matches:")
            for it in enriched:
                print(f"{int(it['rank']):04d}. name={it['name']!r} | fclass={it['fclass']} | lat={it['lat']:.6f}, lon={it['lon']:.6f}")
                if it.get("address"):
                    print(f"      reverse_geocode: {it['address']}")
                elif it.get("address_guess"):
                    print(f"      address_guess:   {it['address_guess']}")
                else:
                    print("      address:         (unavailable)")
            analysis = analyze_anchor_ambiguity(enriched, alpha=0.5, eps=0.28)
            print("\nAnchor ambiguity analysis:")
            print(f"- n_candidates={analysis['n_candidates']}")
            print(f"- spatial-only: K={analysis['spatial_only']['K']} H={analysis['spatial_only']['H_bits']:.4f} bits sizes={analysis['spatial_only']['sizes']}")
            print(f"- semantic-only: K={analysis['semantic_only']['K']} H={analysis['semantic_only']['H_bits']:.4f} bits sizes={analysis['semantic_only']['sizes']} ({analysis['semantic_only']['method']})")
            print(f"- hybrid: alpha={analysis['hybrid']['alpha']} eps={analysis['hybrid']['eps']} K={analysis['hybrid']['K']} H={analysis['hybrid']['H_bits']:.4f} bits sizes={analysis['hybrid']['sizes']} ({analysis['hybrid']['method']})")
            print(f"- anchor_ambiguous (H>1 bits): {analysis['anchor_ambiguous']}")


def build_uq_debug_info(
    client: OpenAI,
    parsed: Dict[str, Any],
    city_abbr: str = "CH",
    target_min_prob: Optional[float] = None,
    mode_min_prob: Optional[float] = None,
    threshold_samples_n: int = 10,
    threshold_eps_m: float = 2000.0,
    threshold_temperature: float = 0.9,
    anchor_alpha: float = 0.5,  # deprecated, kept for compatibility
    anchor_eps: float = 0.28,  # deprecated, use anchor_eps_km instead
    anchor_eps_km: float = 5.0,  # Haversine distance threshold in km
    query: Optional[str] = None,
    skip_reverse_geocode: bool = True,  # Skip reverse geocoding for performance
) -> Dict[str, Any]:
    """
    Build UQ debug info from parsed query + UQ results.
    This is the general UQ function (not EVPI-specific).
    Returns debug_info dict with anchor_clusters, threshold_clusters, CP thresholds, etc.
    """
    # Get anchor candidates
    anchor_text = str(parsed.get("anchor_location") or "").strip()
    rows = anchor_candidates_from_db(city_abbr, anchor_text) if anchor_text and anchor_text.lower() != "unknown" else []
    enriched = enrich_candidates(rows, skip_reverse_geocode=skip_reverse_geocode) if rows else []
    anchor_uq = analyze_anchor_ambiguity(enriched, eps_km=float(anchor_eps_km)) if enriched else {"n_candidates": 0, "anchor_ambiguous": False}
    
    # Build anchor clusters using spatial clustering only
    anchor_clusters: List[Dict[str, Any]] = []
    if enriched:
        # Get spatial clustering labels from anchor_uq
        spatial_labels = anchor_uq.get("spatial_only", {}).get("labels", [])
        if not spatial_labels or len(spatial_labels) != len(enriched):
            # Fallback: treat each candidate as its own cluster if labels don't match
            spatial_labels = list(range(len(enriched)))
        
        # Group candidates by cluster label
        cluster_groups: Dict[int, List[Dict[str, Any]]] = {}
        for idx, label in enumerate(spatial_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append({
                "idx": idx,
                "item": enriched[idx],
            })
        
        # Build clusters: one per spatial cluster
        total_candidates = len(enriched)
        for cluster_id, group in sorted(cluster_groups.items()):
            cluster_size = len(group)
            mass = float(cluster_size) / float(total_candidates)
            
            # Use the first candidate in the cluster as representative (or compute centroid)
            # For now, use first candidate's location
            rep = group[0]["item"]
            rep_lat = float(rep.get("lat"))
            rep_lon = float(rep.get("lon"))
            
            # Compute centroid if multiple candidates
            if cluster_size > 1:
                lat_sum = sum(float(item["item"].get("lat", 0)) for item in group)
                lon_sum = sum(float(item["item"].get("lon", 0)) for item in group)
                rep_lat = lat_sum / cluster_size
                rep_lon = lon_sum / cluster_size
            
            # Build display string from representative
            rep_name = str(rep.get("name") or "")
            rep_addr = str(rep.get("address") or rep.get("address_guess") or "")
            display_str = f"{rep_name}"
            if rep_addr:
                display_str += f" — {rep_addr}"
            if cluster_size > 1:
                display_str += f" (and {cluster_size - 1} more)"
            
            anchor_clusters.append({
                "id": f"A{cluster_id}",
                "name": rep_name,
                "address": rep_addr,
                "lat": rep_lat,
                "lon": rep_lon,
                "mass": mass,
                "display_str": display_str.strip(" —"),
                "cluster_size": cluster_size,
            })
    
    # Get target/mode sets with CP thresholds
    cp_min_t, cp_min_m, cp_path = load_cp_min_probs(city_abbr)
    use_min_t = float(cp_min_t if target_min_prob is None else target_min_prob)
    use_min_m = float(cp_min_m if mode_min_prob is None else mode_min_prob)
    target_probs = complete_and_normalize(parsed.get("target_poi_type", {}) or {}, TARGET_TYPES)
    mode_probs = complete_and_normalize(parsed.get("navigation_mode", {}) or {}, NAV_MODES)
    target_set = list(heuristic_set_from_probs(target_probs, use_min_t))
    mode_set = list(heuristic_set_from_probs(mode_probs, use_min_m))
    
    # Renormalize probabilities within sets
    def renorm_in_set(probs: Dict[str, float], labels: List[str]) -> Dict[str, float]:
        if not labels:
            return {}
        s = sum(float(probs.get(l, 0.0)) for l in labels)
        if s <= 1e-12:
            return {l: 1.0 / len(labels) for l in labels}
        return {l: float(probs.get(l, 0.0)) / s for l in labels}
    
    target_probs_in_set = renorm_in_set(target_probs, target_set)
    mode_probs_in_set = renorm_in_set(mode_probs, mode_set)
    
    # Get threshold clusters
    # Always sample thresholds (N=10) regardless of parsed intent
    parsed_threshold_m = parsed.get("travel_threshold_meters")
    # Use provided query or construct a simple query from parsed intent
    query_text = query or f"Find {parsed.get('target_poi_type', {}).get('target', 'emergency service')} near {parsed.get('anchor_location', 'location')}"
    print(f"\n[Threshold sampling] Always sampling {threshold_samples_n} values (temperature={threshold_temperature})...")
    th_dbg = sample_threshold_distribution_with_raw(client, query=query_text, n=threshold_samples_n, temperature=float(threshold_temperature))
    samples = th_dbg.get("parsed", [])
    
    
    cl = cluster_1d_threshold_samples(samples, eps_m=float(threshold_eps_m))
    threshold_clusters: List[Dict[str, Any]] = []
    for c in cl.get("clusters", []):
        kind = str(c.get("kind") or "meters")
        if kind == "null":
            threshold_clusters.append({
                "id": "Tnull",
                "kind": "null",
                "threshold_m": None,
                "mass": float(c["mass"]),
                "display_str": "No limit (true nearest)",
            })
        else:
            m = float(c["center_m"])
            threshold_clusters.append({
                "id": f"T{c['cluster']}",
                "kind": "meters",
                "threshold_m": m,
                "mass": float(c["mass"]),
                "display_str": f"Within ~{m/1000.0:.1f} km" if m >= 1000 else f"Within ~{int(round(m))} m",
            })
    
    # Fallback if no clusters
    if not threshold_clusters:
        t0 = parsed.get("travel_threshold_meters")
        if t0 is None:
            threshold_clusters = [{
                "id": "Tnull",
                "kind": "null",
                "threshold_m": None,
                "mass": 1.0,
                "display_str": "No limit (true nearest)",
            }]
        else:
            threshold_clusters = [{
                "id": "T0",
                "kind": "meters",
                "threshold_m": float(t0),
                "mass": 1.0,
                "display_str": f"Within ~{float(t0)/1000.0:.1f} km" if float(t0) >= 1000 else f"Within ~{int(round(float(t0)))} m",
            }]
    
    debug_info = {
        "anchor_text": anchor_text,
        "anchor_candidates_n": len(enriched),
        "anchor_uq": anchor_uq,
        "target_probs_full": target_probs,
        "mode_probs_full": mode_probs,
        "cp_min_prob_target": use_min_t,
        "cp_min_prob_mode": use_min_m,
        "cp_thresholds_path": cp_path,
        "threshold_sampling_n": int(cl.get("n", 0)),
        "threshold_entropy_bits": float(cl.get("entropy_bits", 0.0) or 0.0),
        "threshold_clusters_raw": cl.get("clusters", []),
        "threshold_dominant": cl.get("dominant"),
        "threshold_samples_raw": th_dbg.get("raw", []),
        "threshold_samples_parsed": th_dbg.get("parsed", []),
        "threshold_sample_errors": th_dbg.get("errors", []),
        "anchor_clusters": anchor_clusters,
        "threshold_clusters": threshold_clusters,
    }
    
    return debug_info


