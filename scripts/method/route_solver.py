

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import duckdb
import numpy as np
import osmnx as ox  # type: ignore
import pandas as pd

import uq
import time
import sys
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIError, RateLimitError

# Import paths from paths.py
import paths

BASE = paths.BASE
SCRIPTS_DIR = paths.SCRIPTS_DIR

# Ensure routing_module imports resolve
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from routing_module.sssp import nearest_node, sssp_dijkstra, reconstruct_path  # type: ignore

# Database and data paths (using BASE from paths.py)
DB_PATH = BASE / "DB" / "DSPP_DB.duckdb"
POI_DIR = BASE / "DB" / "poi_csv"
GRAPH_CACHE_DIR = BASE / "Data" / "graph-cache"


def _haversine_m_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in meters."""
    R = 6_371_000.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2.astype(float))
    lon2r = np.radians(lon2.astype(float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def _target_labels_for_retrieval(target_label: str) -> List[str]:
    """
    Map an abstract target label to one-or-more concrete POI fclass values.
    """
    t = (target_label or "").strip()
    if not t or t.lower() in {"unknown"}:
        return []
    # Backward compatibility: treat "other" as broad emergency services union.
    if t.lower() in {"other", "emergency", "emergency_services", "emergency service", "emergency services"}:
        return ["police", "fire_station", "hospital", "clinic", "doctors", "pharmacy", "shelter", "community_centre", "school"]
    return [t]


@lru_cache(maxsize=8)
def _poi_table(city_abbr: str) -> str:
    table = f"{city_abbr}_POI"
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB not found: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {t[0] for t in con.execute("SHOW TABLES").fetchall()}
        if table not in tables:
            raise KeyError(f"Table not found in DuckDB: {table}")
        return table
    finally:
        con.close()


def retrieve_candidates_duckdb(
    anchor: Any,
    target_label: str,
    mode_label: str,
    threshold_m: float | None,
    *,
    city_abbr: str = "CH",
    max_candidates: int = 120,
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    LLM-based retrieval (DuckDB):
      - Uses LLM to generate SQL query via TEXT_TO_SQL_PROMPT
      - If threshold_m is provided, filter by distance in SQL using ST_Distance_Spheroid
      - Compute distances in SQL and filter before returning results
      - return nearest max_candidates (to keep routing tractable)
    """
    _ = mode_label
    labels = _target_labels_for_retrieval(target_label)
    if not labels:
        return []
    table = _poi_table(city_abbr)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        # Ensure spatial extension is loaded
        try:
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
        except Exception:
            pass  # Already installed or not available
        
        anchor_lat = float(anchor.lat)
        anchor_lon = float(anchor.lon)
        
        # Generate SQL using LLM
        if client is None:
            # Try to get client from demo module
            try:
                from demo import get_client as get_demo_client
                client = get_demo_client()
            except Exception:
                raise ValueError("OpenAI client required. Pass 'client' parameter or ensure demo.get_client() is available.")
        
        # Generate SQL using LLM (no fallback - propagate errors)
        sql = llm_text_to_sql(
            client,
            city_abbr=city_abbr,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            target_label=target_label,
            threshold_m=threshold_m,
            model=model,
        )
        
        # Add LIMIT clause if not present
        sql_upper = sql.upper().strip()
        if "LIMIT" not in sql_upper:
            sql = sql.rstrip(";").strip() + f" LIMIT {max_candidates}"
        
        print(f"\n[SQL Query] LLM-generated SQL:")
        print(f"  SQL: {sql.strip()}")
        
        # Execute SQL (LLM generates SQL with embedded values, not parameterized)
        rows = con.execute(sql).fetchall()
        print(f"  ✓ Execution SUCCESS: {len(rows)} candidates extracted")
    finally:
        con.close()

    if not rows:
        return []

    out: List[Dict[str, Any]] = []
    for row in rows:
        name, fclass, lat, lon, dist_m = row
        pid = f"{name}|{float(lat):.6f}|{float(lon):.6f}|{fclass}"
        out.append(
            {
                "id": pid,
                "name": str(name),
                "fclass": str(fclass),
                "lat": float(lat),
                "lon": float(lon),
                "dist_m": float(dist_m),
            }
        )
    return out


@lru_cache(maxsize=8)
def _load_poi_df(city_abbr: str = "CH") -> pd.DataFrame:
    """
    CSV fallback. Expected columns:
      name, Lat, Long, fclass
    """
    if city_abbr != "CH":
        raise NotImplementedError(f"POI CSV mapping not implemented for city_abbr={city_abbr!r}")
    path = POI_DIR / "Chicago_POI.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Lat": "lat", "Long": "lon"})
    if "id" not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": "id"})
    df["id"] = df["id"].astype(str)
    df["fclass"] = df["fclass"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    return df


def retrieve_candidates_csv(
    anchor: Any,
    target_label: str,
    mode_label: str,
    threshold_m: float | None,
    *,
    city_abbr: str = "CH",
    max_candidates: int = 120,
) -> List[Dict[str, Any]]:
    _ = mode_label
    df = _load_poi_df(city_abbr)
    labels = _target_labels_for_retrieval(target_label)
    if not labels:
        return []
    sub = df[df["fclass"].isin(labels)]
    if sub.empty:
        return []

    dists_m = _haversine_m_vec(float(anchor.lat), float(anchor.lon), sub["lat"].to_numpy(), sub["lon"].to_numpy())
    if threshold_m is not None and float(threshold_m) > 0:
        keep = dists_m <= float(threshold_m)
        sub = sub.loc[keep]
        dists_m = dists_m[keep]
    if sub.empty:
        return []

    order = np.argsort(dists_m)[:max_candidates]
    out: List[Dict[str, Any]] = []
    rows = sub.iloc[order]
    for i, r in enumerate(rows.itertuples(index=False)):
        out.append(
            {
                "id": getattr(r, "id"),
                "name": getattr(r, "name"),
                "lat": float(getattr(r, "lat")),
                "lon": float(getattr(r, "lon")),
                "fclass": getattr(r, "fclass"),
                "dist_m": float(dists_m[order[i]]),
            }
        )
    return out


def _ensure_graph_exists(mode: str, scenario: str) -> None:
    """Auto-generate missing graph files if they don't exist."""
    mode = mode.lower()
    scen = scenario.lower()
    
    if scen == "abnormal":
        if mode == "walk":
            path = GRAPH_CACHE_DIR / "Chicago_walk_abnormal.graphml"
            if path.exists():
                return
            # Need to build it - first ensure normal walk graph exists
            normal_path = GRAPH_CACHE_DIR / "Chicago_walk.graphml"
            if not normal_path.exists():
                _ensure_graph_exists("walk", "normal")
            print(f"[route_solver] Auto-generating missing graph: {path.name}")
            _build_abnormal_walk_graph(normal_path, path)
        else:
            path = GRAPH_CACHE_DIR / "Chicago_drive_abnormal.graphml"
            if path.exists():
                return
            # Need to build it - first ensure normal drive graph exists
            normal_path = GRAPH_CACHE_DIR / "Chicago_drive.graphml"
            if not normal_path.exists():
                raise FileNotFoundError(f"Cannot build abnormal graph: normal graph not found: {normal_path}")
            print(f"[route_solver] Auto-generating missing graph: {path.name}")
            _build_abnormal_drive_graph(normal_path, path)
    else:
        if mode == "walk":
            path = GRAPH_CACHE_DIR / "Chicago_walk.graphml"
            if path.exists():
                return
            print(f"[route_solver] Auto-generating missing graph: {path.name}")
            _build_normal_walk_graph(path)
        else:
            path = GRAPH_CACHE_DIR / "Chicago_drive.graphml"
            if path.exists():
                return
            print(f"[route_solver] Auto-generating missing graph: {path.name}")
            _build_normal_drive_graph(path)


def _load_gen_module():
    """Lazy import of generate_true_intent module."""
    import sys
    import importlib.util
    gen_path = SCRIPTS_DIR / "data-gen" / "generate_true_intent.py"
    if str(gen_path) not in sys.modules:
        spec = importlib.util.spec_from_file_location("generate_true_intent", gen_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load generate_true_intent from {gen_path}")
        gen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_module)
        sys.modules[str(gen_path)] = gen_module
        return gen_module
    return sys.modules[str(gen_path)]


def _build_normal_drive_graph(output_path: Path) -> None:
    """Build normal drive graph from Chicago polygon."""
    gen_module = _load_gen_module()
    
    city = "Chicago"
    city_poly = gen_module.load_city_polygon(city)
    G = ox.graph_from_polygon(city_poly, network_type="drive")
    
    # Sanitize travel_time
    gen_module.sanitize_travel_time(G)
    
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, output_path)
    print(f"[route_solver] Saved normal drive graph: {output_path}")


def _build_normal_walk_graph(output_path: Path) -> None:
    """Build normal walk graph from Chicago polygon."""
    gen_module = _load_gen_module()
    
    city = "Chicago"
    city_poly = gen_module.load_city_polygon(city)
    G = ox.graph_from_polygon(city_poly, network_type="walk")
    
    # Compute walk_time for all edges
    walk_speed_mps = 1.4
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        data["walk_time"] = float(length_m / walk_speed_mps) if length_m > 0 else 1.0
    
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, output_path)
    print(f"[route_solver] Saved normal walk graph: {output_path}")


def _build_abnormal_drive_graph(normal_path: Path, output_path: Path) -> None:
    """Build abnormal drive graph from normal drive graph using FEMA zones."""
    gen_module = _load_gen_module()
    
    G_normal = ox.load_graphml(normal_path)
    city = "Chicago"
    city_poly = gen_module.load_city_polygon(city)
    
    G_abnormal, stats = gen_module.build_abnormal_graph_fema_simple(G_normal, city_poly, city)
    
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G_abnormal, output_path)
    print(f"[route_solver] Saved abnormal drive graph: {output_path}")
    print(f"[route_solver]   FEMA stats: {stats.get('infinite_H')} infinite edges, {stats.get('scaled_else')} scaled edges")


def _build_abnormal_walk_graph(normal_path: Path, output_path: Path) -> None:
    """Build abnormal walk graph from normal walk graph using FEMA zones."""
    gen_module = _load_gen_module()
    
    G_normal = ox.load_graphml(normal_path)
    city = "Chicago"
    city_poly = gen_module.load_city_polygon(city)
    
    G_abnormal, stats = gen_module.build_abnormal_graph_fema_walk(G_normal, city_poly, city)
    
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G_abnormal, output_path)
    print(f"[route_solver] Saved abnormal walk graph: {output_path}")
    print(f"[route_solver]   FEMA stats: {stats.get('infinite_H')} infinite edges, {stats.get('unchanged')} unchanged edges")


@lru_cache(maxsize=4)
def _load_graph(mode_label: str = "drive", scenario: str = "normal") -> Tuple["object", str]:
    """
    Load cached graphml and return (graph, weight_key).
    
    Now supports proper walk mode:
    - Drive mode: uses drive network, travel_time weight
      - Abnormal: HIGH zones → inf, else 2x delay
    - Walk mode: uses walk network, walk_time weight
      - Abnormal: HIGH zones → inf, NO delays (unchanged)
    
    Auto-generates missing graphs if they don't exist.
    """
    mode = (mode_label or "drive").lower()
    scen = (scenario or "normal").lower()

    if scen == "abnormal":
        if mode == "walk":
            path = GRAPH_CACHE_DIR / "Chicago_walk_abnormal.graphml"
            weight_key = "walk_time"
        else:
            path = GRAPH_CACHE_DIR / "Chicago_drive_abnormal.graphml"
            weight_key = "travel_time"
    else:
        if mode == "walk":
            path = GRAPH_CACHE_DIR / "Chicago_walk.graphml"
            weight_key = "walk_time"
        else:
            path = GRAPH_CACHE_DIR / "Chicago_drive.graphml"
            weight_key = "travel_time"
    
    # Auto-generate if missing
    if not path.exists():
        _ensure_graph_exists(mode, scen)
    
    if not path.exists():
        raise FileNotFoundError(f"Graph cache not found after auto-generation attempt: {path}")

    G = ox.load_graphml(path)
    
    # For normal walk graphs, ensure walk_time exists
    if scen == "normal" and mode == "walk":
        walk_speed_mps = 1.4
        for _u, _v, _k, data in G.edges(keys=True, data=True):
            if "walk_time" not in data or data.get("walk_time") is None:
                length_m = float(data.get("length", 0.0))
                data["walk_time"] = float(length_m / walk_speed_mps) if length_m > 0 else 1.0
    
    return G, weight_key


@lru_cache(maxsize=4096)
def _nearest_node_cached(mode_label: str, scenario: str, lat_round: int, lon_round: int) -> int:
    lat = lat_round / 1e6
    lon = lon_round / 1e6
    G, _w = _load_graph(mode_label, scenario)
    return nearest_node(G, lat, lon)


def _nearest_node(mode_label: str, scenario: str, lat: float, lon: float) -> int:
    return _nearest_node_cached(mode_label, scenario, int(round(lat * 1e6)), int(round(lon * 1e6)))


def _sanitize_graph_weights(G: "object", weight_key: str) -> None:
    """Ensure all edges have valid numeric weights for the specified key."""
    for u, v, k, data in G.edges(keys=True, data=True):
        weight = data.get(weight_key)
        if weight is None:
            continue
        try:
            data[weight_key] = float(weight)
        except (ValueError, TypeError):
            # If conversion fails, set a default based on length
            length = float(data.get("length", 0.0))
            if weight_key == "walk_time":
                data[weight_key] = float(length / 1.4 if length > 0 else 1.0)
            elif weight_key == "travel_time":
                data[weight_key] = float(length / 13.4 if length > 0 else 1.0)
            else:
                data[weight_key] = 1.0


@lru_cache(maxsize=256)
def _sssp_cached(mode_label: str, scenario: str, source_node: int) -> Tuple[Dict[int, float], Dict[int, list]]:
    """Cache SSSP results: returns (distances, paths)."""
    G, weight = _load_graph(mode_label, scenario)
    # Sanitize weights before routing to ensure all are numeric
    _sanitize_graph_weights(G, weight)
    distances, paths = sssp_dijkstra(G, source_node, weight=weight)
    return {int(k): float(v) for k, v in distances.items()}, {int(k): list(v) for k, v in paths.items()}


def route_cost_graph(
    anchor: Any,
    poi: Dict[str, Any],
    mode_label: str,
    *,
    scenario: str = "normal",
) -> float:
    """
    Mode-aware routing using cached Chicago graphml:
      - drive: shortest path travel_time seconds
      - walk: shortest path walk_time seconds
    """
    mode = (mode_label or "drive").lower()
    scen = (scenario or "normal").lower()
    
    # Debug: verify graph and weight being used
    try:
        G, weight = _load_graph(mode, scen)
    except Exception as e:
        print(f"    ✗ Failed to load graph (mode={mode}, scenario={scen}): {e}")
        return float("inf")
    
    graph_path = GRAPH_CACHE_DIR / f"Chicago_{mode}_{scen}.graphml" if scen == "abnormal" else GRAPH_CACHE_DIR / f"Chicago_{mode}.graphml"
    
    try:
        src = _nearest_node(mode, scenario, float(anchor.lat), float(anchor.lon))
        lat = float(poi.get("lat"))
        lon = float(poi.get("lon"))
        tgt = _nearest_node(mode, scenario, lat, lon)
        
        # Check if nodes are valid
        if src not in G.nodes() or tgt not in G.nodes():
            if not hasattr(route_cost_graph, '_node_warned'):
                route_cost_graph._node_warned = set()
            key = (mode, scenario, round(float(anchor.lat), 2), round(float(anchor.lon), 2))
            if key not in route_cost_graph._node_warned and len(route_cost_graph._node_warned) < 2:
                route_cost_graph._node_warned.add(key)
                print(f"    ⚠ Invalid nodes: src={src} (in graph: {src in G.nodes()}), tgt={tgt} (in graph: {tgt in G.nodes()})")
            return float("inf")
        
        distances, _paths = _sssp_cached(mode, scenario, src)
        cost = float(distances.get(int(tgt), float("inf")))
        
        # Debug: log if infinite (first few times)
        if cost == float("inf"):
            if not hasattr(route_cost_graph, '_inf_warned'):
                route_cost_graph._inf_warned = set()
            key = (mode, scenario, round(float(anchor.lat), 3), round(float(anchor.lon), 3))
            if key not in route_cost_graph._inf_warned and len(route_cost_graph._inf_warned) < 2:
                route_cost_graph._inf_warned.add(key)
                print(f"    ⚠ Route cost infinite: anchor=({float(anchor.lat):.4f}, {float(anchor.lon):.4f}), poi=({lat:.4f}, {lon:.4f})")
                print(f"       src_node={src}, tgt_node={tgt}, path_exists={tgt in distances}")
        
        return cost
    except FileNotFoundError as e:
        print(f"    ✗ Graph file missing: {e}")
        return float("inf")
    except Exception as e:
        if not hasattr(route_cost_graph, '_error_warned'):
            route_cost_graph._error_warned = set()
        error_key = str(type(e).__name__)
        if error_key not in route_cost_graph._error_warned:
            route_cost_graph._error_warned.add(error_key)
            print(f"    ✗ Routing error ({error_key}): {e}")
        return float("inf")


@lru_cache(maxsize=256)
def _route_distance_cached(mode_label: str, scenario: str, source_node: int, target_node: int) -> float:
    """Compute actual route distance in meters along shortest path."""
    G, weight = _load_graph(mode_label, scenario)
    distances, paths = _sssp_cached(mode_label, scenario, source_node)
    
    if target_node not in distances:
        return float("inf")
    
    path = reconstruct_path(paths, source_node, target_node)
    if not path or len(path) < 2:
        return float("inf")
    
    total_length = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        # Get edge with minimum weight (in case of parallel edges)
        edge_data = min(G[u][v].values(), key=lambda d: d.get(weight, float("inf")))
        length_m = float(edge_data.get("length", 0.0))
        total_length += length_m
    return float(total_length)


def route_distance_graph(
    anchor: Any,
    poi: Dict[str, Any],
    mode_label: str,
    *,
    scenario: str = "normal",
) -> float:
    """
    Compute actual route distance in meters along shortest path.
    """
    mode = (mode_label or "drive").lower()
    src = _nearest_node(mode, scenario, float(anchor.lat), float(anchor.lon))
    lat = float(poi.get("lat"))
    lon = float(poi.get("lon"))
    tgt = _nearest_node(mode, scenario, lat, lon)
    return _route_distance_cached(mode, scenario, src, tgt)


def route_cost_haversine(anchor: Any, poi: Dict[str, Any], mode_label: str) -> float:
    """
    Deterministic fallback: straight-line distance (meters).
    """
    _ = mode_label
    lat = float(poi.get("lat") if poi.get("lat") is not None else poi.get("Lat"))
    lon = float(poi.get("lon") if poi.get("lon") is not None else poi.get("Long"))
    return float(uq.haversine_km(float(anchor.lat), float(anchor.lon), lat, lon) * 1000.0)


def _extract_sql(text: str) -> str:
    s = (text or "").strip()
    # strip code fences if any
    if s.startswith("```"):
        s = "\n".join([ln for ln in s.splitlines() if not ln.strip().startswith("```")]).strip()
    
    # Find the first SELECT statement (case-insensitive)
    import re
    select_match = re.search(r'(?i)\bSELECT\b', s)
    if not select_match:
        # No SELECT found, return as-is
        return s.strip()
    
    # Extract from SELECT onwards
    sql_start = select_match.start()
    sql_text = s[sql_start:]
    
    # Find the first semicolon that ends the SQL statement
    # Look for semicolon followed by whitespace or end of string, or followed by non-SQL text
    semicolon_match = re.search(r';\s*(?:\n|$|Human|User|Assistant|parsed_query|ref_info)', sql_text, re.IGNORECASE)
    if semicolon_match:
        # Extract up to and including the semicolon
        sql_text = sql_text[:semicolon_match.end()].rstrip()
        # Remove any trailing text after semicolon
        sql_text = sql_text.split(';')[0] + ';'
    else:
        # No semicolon found, try to find end of SQL by looking for common prompt patterns
        # Stop at common prompt indicators
        for pattern in [r'\n\s*Human:', r'\n\s*User:', r'\n\s*Assistant:', r'\n\s*parsed_query', r'\n\s*ref_info']:
            match = re.search(pattern, sql_text, re.IGNORECASE)
            if match:
                sql_text = sql_text[:match.start()].strip()
                # Add semicolon if not present
                if not sql_text.endswith(';'):
                    sql_text += ';'
                break
    
    return sql_text.strip()


def llm_text_to_sql(
    client: OpenAI,
    *,
    city_abbr: str,
    anchor_lat: float | None,
    anchor_lon: float | None,
    target_label: str | None,
    threshold_m: float | None,
    model: str = "gpt-4o-mini",
    max_retries: int = 6,
) -> str:
    """
    Generate DuckDB SQL via LLM (for logging/debug).

    Table schema:
      <CITY>_POI(name VARCHAR, lat DOUBLE, long DOUBLE, fclass VARCHAR, geom GEOMETRY)

    Requirements:
      - output RAW SQL only
      - select name,fclass,lat,long, and euclid_distance_m (meters) as computed column
      - filter by fclass if target_label known
      - if threshold_m is not None: filter by distance <= threshold_m
      - do not sort
    """
    table = f"{city_abbr}_POI"
    labels = _target_labels_for_retrieval(target_label)
    # If we have a broad "other/emergency services", the LLM should use IN (...)
    fclass_filter = labels
    import prompts
    
    # Format for old prompt style: parsed_query and ref_info
    # If multiple labels (e.g., emergency services), pass as list
    target_poi_type = labels if len(labels) > 1 else (labels[0] if labels else None)
    parsed_query = {
        "start_location": None,  # Not used in current implementation
        "target_poi_type": target_poi_type,
        "distance_threshold": None if threshold_m is None else float(threshold_m),
        "navigation_mode": None,  # Not used in SQL generation
    }
    ref_info = {
        "name": None,  # Anchor name not available here
        "lat": float(anchor_lat) if anchor_lat is not None else None,
        "long": float(anchor_lon) if anchor_lon is not None else None,
        "fclass": None,  # Not used
    }
    
    prompt = prompts.TEXT_TO_SQL_PROMPT.format(
        table=table,
        parsed_query_json_here=json.dumps(parsed_query, ensure_ascii=False),
        ref_info_json_here=json.dumps(ref_info, ensure_ascii=False)
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            # Handle Mistral response format (dict) vs OpenAI format (object)
            if isinstance(resp, dict):
                content = resp.get("choices", [{}])[0].get("message", {}).get("content") or ""
            else:
                content = resp.choices[0].message.content or ""
            print('raw sql', content)
            return _extract_sql(content)
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            last_err = e
            time.sleep(min(2.0**attempt, 10.0))
    raise RuntimeError(f"llm_text_to_sql failed after {max_retries} retries: {last_err}")


def default_retrieve(anchor_lat: float, anchor_lon: float, target: str, mode: str, threshold_m: Optional[float], city_abbr: str = "CH", client: Optional[OpenAI] = None) -> List[Dict[str, Any]]:
    """Wrapper for existing retrieve_candidates_duckdb (general utility, not EVPI-specific).
    No fallback - errors are propagated.
    """
    from dataclasses import dataclass as dc
    
    @dc
    class FakeAnchor:
        lat: float
        lon: float
    
    fake_anchor = FakeAnchor(lat=anchor_lat, lon=anchor_lon)
    # No fallback - propagate errors if SQL generation/execution fails
    return retrieve_candidates_duckdb(fake_anchor, target, mode, threshold_m, city_abbr=city_abbr, client=client)


