
"""
Dataset loaders for GT city
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import paths from paths.py for consistency
import paths
BASE = paths.BASE
DATA_DIR = BASE / "Data"

# Default to Chicago, but can be overridden
DEFAULT_CITY = "Chicago"


def city_slug(city_name: str) -> str:
    """Convert city name to folder-friendly slug (e.g., 'Chicago' -> 'Chicago', 'New York' -> 'New_York')."""
    return city_name.replace(" ", "_")


def get_gt_dir(city: str = DEFAULT_CITY) -> Path:
    """Get GT directory for a city (new convention uses full city name)."""
    city_slug_name = city_slug(city)
    return DATA_DIR / f"GT_{city_slug_name}"


@lru_cache(maxsize=4)
def load_query(city: str = DEFAULT_CITY) -> List[Dict[str, Any]]:
    """Load query file for a city."""
    city_slug_name = city_slug(city)
    path = get_gt_dir(city) / f"GT_{city_slug_name}_query.json"
    if not path.exists():
        raise SystemExit(f"Query file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_gt_poi_abnormal(city: str = DEFAULT_CITY) -> List[Dict[str, Any]]:
    """Load GT POI abnormal file for a city.

    Supports both the old and new naming conventions:
      - Old (Chicago-style): Data/GT_Chicago/GT_Chicago_POI.json
      - New (city-wise abnormal): Data/GT_HO/HO_gt_poi_abnormal.json, etc.
    """
    # First, try the "old" convention based on full city name (e.g., Chicago)
    city_slug_name = city_slug(city)
    old_path = get_gt_dir(city) / f"GT_{city_slug_name}_POI.json"
    if old_path.exists():
        return json.loads(old_path.read_text(encoding="utf-8"))

    # Next, try the new city-wise abnormal convention based on abbreviation
    # e.g., HO -> Data/GT_HO/HO_gt_poi_abnormal.json
    # or Houston -> HO (via inverse mapping).
    abbr = city_name_to_abbr(city)
    if abbr:
        new_dir = DATA_DIR / f"GT_{abbr}"
        new_path = new_dir / f"{abbr}_gt_poi_abnormal.json"
        if new_path.exists():
            return json.loads(new_path.read_text(encoding="utf-8"))

    # If neither path exists, raise a clear error
    raise SystemExit(
        f"GT POI file not found for city '{city}'. "
        f"Tried: {old_path} and (if applicable) {DATA_DIR / f'GT_{abbr}' / f'{abbr}_gt_poi_abnormal.json'}"
    )


# Backward compatibility aliases
@lru_cache(maxsize=4)
def load_ch_query() -> List[Dict[str, Any]]:
    """Backward compatibility: load Chicago query."""
    return load_query("Chicago")


@lru_cache(maxsize=4)
def load_ch_gt_poi_abnormal() -> List[Dict[str, Any]]:
    """Backward compatibility: load Chicago GT POI."""
    return load_gt_poi_abnormal("Chicago")


@lru_cache(maxsize=8)
def index_by_query_id(items_json: str) -> Dict[str, Dict[str, Any]]:
    items = json.loads(items_json)
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for obj in items:
            qid = str(obj.get("query_id") or obj.get("id") or "").strip()
            if qid:
                out[qid] = obj
    return out


def get_record_by_query_id(query_id: str, city: str = DEFAULT_CITY) -> Optional[Dict[str, Any]]:
    """Get record by query_id for a city."""
    qid = str(query_id).strip()
    if not qid:
        return None
    data = load_query(city)
    idx = index_by_query_id(json.dumps(data))
    return idx.get(qid)


def city_abbr_to_name(city_abbr: str) -> str:
    """Convert city abbreviation to full city name."""
    abbr_map = {
        "CH": "Chicago",
        "NY": "New York",
        "SA": "San Antonio",
        "NO": "New Orleans",
        "MI": "Miami",
        "HO": "Houston",
    }
    return abbr_map.get(city_abbr.upper(), city_abbr)


def city_name_to_abbr(city_name: str) -> str:
    """Convert full city name to abbreviation, if known."""
    name_map = {
        "Chicago": "CH",
        "New York": "NY",
        "San Antonio": "SA",
        "New Orleans": "NO",
        "Miami": "MI",
        "Houston": "HO",
    }
    # Also allow passing an abbreviation directly
    if city_name in name_map.values():
        return city_name
    return name_map.get(city_name)


def get_gt_abnormal_by_query_id(query_id: str, city: str = DEFAULT_CITY) -> Optional[Dict[str, Any]]:
    """Get GT abnormal POI by query_id for a city.
    
    Args:
        query_id: Query ID
        city: City name (e.g., "Chicago") or city abbreviation (e.g., "CH")
    """
    qid = str(query_id).strip()
    if not qid:
        return None
    
    # Convert city abbreviation to full name if needed
    city_name = city_abbr_to_name(city)
    
    try:
        data = load_gt_poi_abnormal(city_name)
        idx = index_by_query_id(json.dumps(data))
        return idx.get(qid)
    except SystemExit:
        # File doesn't exist, return None instead of crashing
        return None




