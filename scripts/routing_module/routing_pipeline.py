

import os
import json
import csv
import math
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import duckdb
import pandas as pd
import time
import json as _json

from .routing_core import map_points_to_nodes
from .sssp import nearest_node, sssp_dijkstra, reconstruct_path

# ------------ Config ------------
BASE = Path("/home/ubuntu/DSPP")
DB_FILE = BASE / "DB" / "DSPP_DB.duckdb"
STATIC_INFO_DIR = BASE / "Result-Ablation/GPT-4/LLM"
OUTPUT_DIR = BASE / "Result-Ablation/GPT-4/Routing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEMA_SHP = BASE / "shp" / "City_HZ_AREAS.shp"
CITY_POLY_SHP = BASE / "shp" / "City_Poly.shp"
FACTOR_HIGH = float("inf")  # simulate closure with infinity
FACTOR_ELSE = 2.0

# ------------ Config Variables (change these each run) ------------
CITY_NAME = "Chicago"  # Change this: e.g., "Chicago", "Miami", "New Orleans", etc.
TABLE_NAME = "CH_POI"  # Change this: e.g., "CH_POI", "MI_POI", "NO_POI", etc.
SAVE_LEGACY_SUMMARIES = True  # If False, skip writing summary/time CSVs and keep only llm_sql_debug.csv
MAX_QUERIES: Optional[int] = None  
CITY_SLUG = CITY_NAME.replace(" ", "_")
STATIC_INFO_FILE = STATIC_INFO_DIR / f"{CITY_SLUG}_static_info_gpt4.json"



def require_gis():
    """Import GIS packages."""
    try:
        import geopandas as gpd  
        from shapely.prepared import prep  
        from shapely.ops import unary_union  
    except Exception as e:
        raise SystemExit(f"GIS packages required: {e}")
    return gpd, prep, unary_union


class _NullDictWriter:
    def writeheader(self) -> None:
        pass
    def writerow(self, row) -> None:
        pass


def _find_labels_file(city_name: str) -> Optional[Path]:
    """
    Locate abnormal_allpoi labels for a city under GT_Labels, handling space/underscore variants.
    """
    base = BASE / "GT_Labels"
    city_dir = base / city_name
    if not city_dir.exists():
        return None
    slug_underscore = city_name.lower().replace(" ", "_")
    slug_space = city_name.lower()
    candidates = [
        city_dir / f"{slug_underscore}_labels_abnormal_allpoi.json",
        city_dir / f"{slug_space}_labels_abnormal_allpoi.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_labels_map(city_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load abnormal labels and index by id.
    """
    p = _find_labels_file(city_name)
    if not p:
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return {}
    except Exception:
        return {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for obj in data:
        qid = str(obj.get("id") or "").strip()
        if qid:
            by_id[qid] = obj
    return by_id


def _find_gt_sql_file(city_name: str) -> Optional[Path]:
    """
    Locate GT SQL candidates JSON produced by generate_label_sql.py under GT_SQL/<City>/<City>_sql_candidates.json
    """
    base = BASE / "GT_SQL"
    city_dir = base / city_name
    if not city_dir.exists():
        return None
    slug_underscore = city_name.replace(" ", "_")
    p = city_dir / f"{slug_underscore}_sql_candidates.json"
    return p if p.exists() else None


def _load_gt_sql_map(city_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Load GT SQL results and index by id.
    Each entry contains fields like: GT_Query, GT_Count, GT_Candidates, distance_threshold_m, etc.
    """
    p = _find_gt_sql_file(city_name)
    if not p:
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return {}
    except Exception:
        return {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for obj in data:
        qid = str(obj.get("id") or "").strip()
        if qid:
            by_id[qid] = obj
    return by_id


def sanitize_travel_time(G: "object", fallback_speed_mps: float = 13.4) -> None:
    """Ensure all edges have valid travel_time."""
    for u, v, k, data in G.edges(keys=True, data=True):
        tt = data.get("travel_time")
        if tt is None or float(tt) <= 0.0:
            length = float(data.get("length", 0.0))
            data["travel_time"] = float(length / fallback_speed_mps if length > 0 else 1.0)


def fema_zone_classifier(zone_code: str) -> str:
    """Classify FEMA FLD_ZONE into HIGH/MED/NONE."""
    if not zone_code:
        return "NONE"
    z = str(zone_code).strip().upper()
    if z in {"A", "AE", "AH", "AO", "V", "VE"}:
        return "HIGH"
    if z in {"X", "0.2", "B", "C"}:
        return "MED"
    return "NONE"

def build_route_geometry_lonlat(
    G,
    path,
    weight_key: str = "travel_time",
    length_key: str = "length",
):
    """
    Concatenate per-edge geometry for a route.

    Returns:
        coords: [[lon, lat], ...] for GeoJSON LineString
        total_length_m: float, sum of edge `length` attributes
    """
    if not path or len(path) < 2:
        return [], 0.0

    coords = []
    total_length_m = 0.0

    for u, v in zip(path[:-1], path[1:]):
        data_dict = G.get_edge_data(u, v)
        if not data_dict:
            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
            if not coords:
                coords.append([ux, uy])
            if coords[-1] != [vx, vy]:
                coords.append([vx, vy])
            continue

        
        edge_data = min(data_dict.values(), key=lambda d: float(d.get(weight_key, 1e9)))

       
        total_length_m += float(edge_data.get(length_key, 0.0))

        geom = edge_data.get("geometry")

        if geom is not None:
            try:
                seq = list(geom.coords)
            except Exception:
                try:
                    from shapely.ops import linemerge
                    seq = list(linemerge(geom).coords)
                except Exception:
                    seq = []

            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            if seq:
                def manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
                if manhattan(seq[0], (ux, uy)) > manhattan(seq[-1], (ux, uy)):
                    seq = seq[::-1]

                if not coords:
                    coords.append([seq[0][0], seq[0][1]])
                for x, y in seq[1:]:
                    if coords[-1][0] != x or coords[-1][1] != y:
                        coords.append([x, y])
            else:
                vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
                if not coords:
                    coords.append([ux, uy])
                if coords[-1] != [vx, vy]:
                    coords.append([vx, vy])
        else:
            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
            if not coords:
                coords.append([ux, uy])
            if coords[-1] != [vx, vy]:
                coords.append([vx, vy])

    return coords, total_length_m

def load_city_polygon_from_master(city_name: str):
    """Load city polygon from City_Poly.shp."""
    gpd, _, unary_union = require_gis()
    if not CITY_POLY_SHP.exists():
        return None
    gdf = gpd.read_file(str(CITY_POLY_SHP))
    gdf.columns = [c.strip() for c in gdf.columns]
    name_col = next((c for c in gdf.columns if c.lower() == "name"), None)
    if name_col:
        sub = gdf[gdf[name_col].astype(str).str.strip().str.lower() == city_name.lower()]
        if sub.empty:
            return None
        sub = sub.to_crs(epsg=4326)
        return unary_union(sub.geometry)
    return None


def build_abnormal_graph_fema_simple(G: "object", city_polygon, shp_path: Path, city_name: str):
    """Build FEMA abnormal graph: HIGH=infinity, ELSE=2.0x."""
    import osmnx as ox  # type: ignore
    gpd, prep, unary_union = require_gis()
    if not shp_path.exists():
        raise SystemExit(f"FEMA shapefile not found: {shp_path}")
    zones = gpd.read_file(str(shp_path))
    zones.columns = [c.strip() for c in zones.columns]
    ccol = next((c for c in zones.columns if c.lower() in {"city", "name", "city_name"}), None)
    if ccol:
        zones = zones[zones[ccol].astype(str).str.strip().str.lower() == city_name.lower()].copy()
    zcol = next((c for c in zones.columns if c.lower() in {"fld_zone", "fldzone", "zone", "flood_zone"}), None)
    if zcol is None:
        raise SystemExit("FEMA shapefile missing FLD_ZONE-like column.")
    zones["RISK"] = zones[zcol].apply(fema_zone_classifier)
    high_union = unary_union(zones[zones["RISK"] == "HIGH"].geometry) if not zones[zones["RISK"] == "HIGH"].empty else None
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    edges_3857 = edges_gdf.to_crs(epsg=3857)
    H = None
    if high_union is not None:
        high_3857 = gpd.GeoSeries([high_union], crs=zones.crs).to_crs(epsg=3857).iloc[0]
        H = prep(high_3857)
    G_abn = G.copy()
    sanitize_travel_time(G_abn)
    stats = {"total_edges": int(edges_3857.shape[0]), "infinite_H": 0, "scaled_else": 0}
    for (u, v, k), row in edges_3857.iterrows():
        if not G_abn.has_edge(u, v, k):
            continue
        data = G_abn.get_edge_data(u, v, k)
        if data is None:
            continue
        tt = data.get("travel_time")
        if tt is None or float(tt) <= 0.0:
            length = float(data.get("length", 0.0))
            tt = (length / 13.4 if length > 0 else 1.0)
        geom = row.geometry
        if H is not None and (H.contains(geom.centroid) or H.intersects(geom)):
            data["travel_time"] = float("inf")
            stats["infinite_H"] += 1
        else:
            data["travel_time"] = float(tt) * FACTOR_ELSE
            stats["scaled_else"] += 1
    return G_abn, stats


def pick_top_k_routes(G_sub, source_node: int, cand_nodes: List[int], k_min: int = 3, k_max: int = 5):
    """Run SSSP and return top-K routes by travel_time."""
    seen = set()
    uniq = []
    for n in cand_nodes:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    distances, paths = sssp_dijkstra(G_sub, source_node, weight="travel_time")
    results = []
    for cn in uniq:
        if cn not in distances:
            continue
        tt = float(distances[cn])
        if not math.isfinite(tt):
            continue
        path = reconstruct_path(paths, source_node, cn)
        if not path or len(path) < 2 or tt <= 0.0:
            continue
        results.append((cn, tt, path))
    results.sort(key=lambda x: x[1])
    if len(results) < k_min:
        return results
    return results[: min(k_max, len(results))]


def process_routing(
    static_info: Dict[str, Any],
    G: "object",
    G_abnormal: "object",
    con: duckdb.DuckDBPyConnection,
    labels_map: Dict[str, Dict[str, Any]],
    output_file_handle,
    time_summary_writer,
    csv_writer,
    xl_writer,
    routes_collector: Dict[str, Any]
) -> Dict[str, Any]:
    """Process one query: execute SQL, route, return result."""
    qid = static_info["id"]
    query = static_info["query"]
    parsed = static_info.get("parsed")
    ref_info = static_info.get("ref_info")
    sql = static_info.get("sql")
    t_parse_sec = static_info.get("t_parse_sec")
    t_sql_gen_sec = static_info.get("t_sql_gen_sec")
    
    routing_start = time.perf_counter()
    
    sql_executed_ok = False
    sql_error = None
    candidates_sql = []
    candidate_count = 0
    t_sql_exec_sec = None
    
    if sql:
        try:
            t_sql_start = time.perf_counter()
            rows = con.execute(sql).fetchall()
            t_sql_exec_sec = time.perf_counter() - t_sql_start
            sql_executed_ok = True
            candidates: List[Dict[str, Any]] = []
            for r in rows:
                try:
                    if len(r) < 3:
                        continue
                    if len(r) >= 4:
                        name_val = r[0]
                        fclass_val = r[1]
                        lat_val = float(r[2]) if r[2] is not None else None
                        lon_val = float(r[3]) if r[3] is not None else None
                        euclid_val = float(r[4]) if len(r) > 4 and r[4] is not None else None
                    else:
                        name_val = r[0]
                        fclass_val = None
                        lat_val = float(r[1]) if r[1] is not None else None
                        lon_val = float(r[2]) if r[2] is not None else None
                        euclid_val = float(r[3]) if len(r) > 3 and r[3] is not None else None
                    candidates.append({
                        "name": name_val,
                        "fclass": fclass_val,
                        "lat": lat_val,
                        "long": lon_val,
                        "euclid_distance_m": euclid_val
                    })
                except Exception:
                    continue
            candidates_sql = candidates
            candidate_count = len(candidates_sql)
        except Exception as e:
            sql_error = str(e)
            sql_executed_ok = False
    
    top5_normal = []
    top5_abnormal = []
    t_node_mapping_sec = None
    t_routing_normal_sec = None
    t_routing_abnormal_sec = None
    
    if sql_executed_ok and ref_info and candidate_count > 0:
        ref_lat = ref_info["lat"]
        ref_lon = ref_info["long"]
        
        t_node_map_start = time.perf_counter()
        all_lats = [ref_lat] + [c["lat"] for c in candidates_sql]
        all_lons = [ref_lon] + [c["long"] for c in candidates_sql]
        all_nodes = map_points_to_nodes(G, all_lats, all_lons)
        src_node = all_nodes[0] if all_nodes else None
        cand_nodes_raw = all_nodes[1:] if len(all_nodes) > 1 else []
        t_node_mapping_sec = time.perf_counter() - t_node_map_start
        
        node_to_cand_idx: Dict[int, int] = {}
        cand_nodes_for_route: List[int] = []
        for idx, node in enumerate(cand_nodes_raw):
            if node is None:
                continue
            cand_nodes_for_route.append(node)
            node_to_cand_idx[node] = idx

        if src_node is None:
            print(f"[{qid}] WARNING: Could not map source to graph node")
        elif cand_nodes_for_route:
            try:
                t_route_n_start = time.perf_counter()
                routes_n = pick_top_k_routes(G, src_node, cand_nodes_for_route, k_min=1, k_max=5)
                t_routing_normal_sec = time.perf_counter() - t_route_n_start
                
                for node, tt, path in routes_n:
                    cand_idx = node_to_cand_idx.get(node)
                    cand = candidates_sql[cand_idx] if (cand_idx is not None and cand_idx < len(candidates_sql)) else None
                    
                    geom_coords, length_m = build_route_geometry_lonlat(G, path)
                    
                    top5_normal.append({
                        "candidate_name": cand["name"] if cand else None,
                        "candidate_fclass": cand.get("fclass") if cand else None,
                        "candidate_lat": cand["lat"] if cand else None,
                        "candidate_lon": cand["long"] if cand else None,
                        "travel_time_s": tt,
                        "length_m": length_m,
                        "euclid_distance_m": cand.get("euclid_distance_m") if cand else None,
                        "_geometry_lonlat": geom_coords  # Internal use for shapefile, not in JSON
                    })
            except Exception as e:
                print(f"[{qid}] ERROR: Normal routing failed: {e}")
            
            try:
                t_route_a_start = time.perf_counter()
                routes_a = pick_top_k_routes(G_abnormal, src_node, cand_nodes_for_route, k_min=1, k_max=5)
                t_routing_abnormal_sec = time.perf_counter() - t_route_a_start
                
                for node, tt, path in routes_a:
                    cand_idx = node_to_cand_idx.get(node)
                    cand = candidates_sql[cand_idx] if (cand_idx is not None and cand_idx < len(candidates_sql)) else None
                    
                    geom_coords, length_m = build_route_geometry_lonlat(G_abnormal, path)
                    top5_abnormal.append({
                        "candidate_name": cand["name"] if cand else None,
                        "candidate_fclass": cand.get("fclass") if cand else None,
                        "candidate_lat": cand["lat"] if cand else None,
                        "candidate_lon": cand["long"] if cand else None,
                        "travel_time_s": tt,
                        "length_m": length_m,
                        "euclid_distance_m": cand.get("euclid_distance_m") if cand else None,
                        "_geometry_lonlat": geom_coords  # Internal use for shapefile, not in JSON
                    })
            except Exception as e:
                print(f"[{qid}] ERROR: Abnormal routing failed: {e}")
    
    pipeline_total_sec = (t_parse_sec or 0) + (t_sql_gen_sec or 0) + (t_sql_exec_sec or 0) + \
                        (t_node_mapping_sec or 0) + (t_routing_normal_sec or 0) + (t_routing_abnormal_sec or 0)
    
    t_total_routing_sec = time.perf_counter() - routing_start
    
    def clean_route_for_json(route):
        """Remove internal geometry field before JSON serialization"""
        cleaned = {k: v for k, v in route.items() if not k.startswith("_")}
        return cleaned
    
    if top5_normal or top5_abnormal:
        key = qid
        routes_collector[key] = {
            "reference": {
                "name": ref_info.get("name") if ref_info else None,
                "latitude": ref_info.get("lat") if ref_info else None,
                "longitude": ref_info.get("long") if ref_info else None,
            },
            "routes_normal": [
                {
                    "to": {"name": r.get("candidate_name")},
                    "metrics": {
                        "time_ff_s": r.get("travel_time_s"),
                        "distance_ff_m": r.get("length_m"),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": r.get("_geometry_lonlat", []),  # Use internal geometry field (still available here)
                    },
                }
                for r in top5_normal  # Save ALL routes from top5_normal (every pred should have a route)
            ],
            "routes_abnormal": [
                {
                    "to": {"name": r.get("candidate_name")},
                    "metrics": {
                        "time_ff_s": r.get("travel_time_s"),
                        "distance_ff_m": r.get("length_m"),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": r.get("_geometry_lonlat", []),  # Use internal geometry field (still available here)
                    },
                }
                for r in top5_abnormal  # Save ALL routes from top5_abnormal (every pred should have a route)
            ],
        }
    
    result = {
        "id": qid,
        "query": query,
        "parsed": parsed,
        "ref_info": ref_info,
        "sql": sql,
        "sql_executed_ok": sql_executed_ok,
        "sql_error": sql_error,
        "candidate_count": candidate_count,
        "top5_normal": [clean_route_for_json(r) for r in top5_normal[:5]],  # Clean AFTER saving to routes_collector
        "top5_abnormal": [clean_route_for_json(r) for r in top5_abnormal[:5]],  # Clean AFTER saving to routes_collector
        "t_parse_sec": t_parse_sec,
        "t_sql_gen_sec": t_sql_gen_sec,
        "t_sql_exec_sec": t_sql_exec_sec,
        "t_node_mapping_sec": t_node_mapping_sec,
        "t_routing_normal_sec": t_routing_normal_sec,
        "t_routing_abnormal_sec": t_routing_abnormal_sec,
        "t_total_routing_sec": t_total_routing_sec,
        "pipeline_total_sec": pipeline_total_sec
    }
    
    # Write to time summary CSV
    time_summary_writer.writerow({
        "query_id": qid,
        "t_parse_sec": t_parse_sec,
        "t_sql_gen_sec": t_sql_gen_sec,
        "t_sql_exec_sec": t_sql_exec_sec,
        "t_node_mapping_sec": t_node_mapping_sec,
        "t_routing_normal_sec": t_routing_normal_sec,
        "t_routing_abnormal_sec": t_routing_abnormal_sec,
        "t_total_routing_sec": t_total_routing_sec,
        "pipeline_total_sec": pipeline_total_sec
    })
    
    csv_writer.writerow({
        "id": qid,
        "query": query,
        "ref_name": ref_info.get("name") if ref_info else None,
        "ref_lat": ref_info.get("lat") if ref_info else None,
        "ref_lon": ref_info.get("long") if ref_info else None,
        "target_category": parsed.get("target_poi_type") if parsed else None,
        "sql": sql,
        "executed_ok": sql_executed_ok,
        "error_detail": sql_error,
        "candidate_count": candidate_count,
        "top1_normal_name": top5_normal[0].get("candidate_name") if top5_normal else None,
        "top1_normal_time_s": top5_normal[0].get("travel_time_s") if top5_normal else None,
        "top1_abnormal_name": top5_abnormal[0].get("candidate_name") if top5_abnormal else None,
        "top1_abnormal_time_s": top5_abnormal[0].get("travel_time_s") if top5_abnormal else None,
    })
    
    gt_candidates_min: List[Dict[str, Any]] = []
    gt_count = None
    gt_sql = None
    try:
        entry = labels_map.get(qid) if labels_map else None
        if entry:
            gt_sql = entry.get("GT_Query")
            cands = entry.get("GT_Candidates") or []
            for c in cands:
                try:
                    gt_candidates_min.append({
                        "name": c.get("name"),
                        "lat": float(c.get("lat")) if c.get("lat") is not None else None,
                        "long": float(c.get("long")) if c.get("long") is not None else None,
                    })
                except Exception:
                    continue
            gt_count = int(entry.get("GT_Count")) if isinstance(entry.get("GT_Count"), (int, float)) else len(gt_candidates_min)
    except Exception:
        gt_candidates_min = []
        gt_count = None

    try:
        xl_writer.writerow({
            "id": qid,
            "query": query,
            "parsed_json": _json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
            "ref_info_json": _json.dumps(ref_info, ensure_ascii=False) if ref_info is not None else None,
            "sql_generated": sql,
            "sql_status_json": _json.dumps({"executed": bool(sql_executed_ok), "error": sql_error}, ensure_ascii=False),
            "pred_candidate_count": candidate_count,
            "pred_candidates_json": _json.dumps(
                [
                    {"name": c.get("name"), "fclass": c.get("fclass"), "lat": c.get("lat"), "long": c.get("long")}
                    for c in (candidates_sql or [])
                ],
                ensure_ascii=False
            ),
            "gt_sql_query": gt_sql,
            "gt_sql_count": gt_count,
            "gt_sql_candidates_json": _json.dumps(gt_candidates_min, ensure_ascii=False),
            "top5_abnormal_json": _json.dumps(
                [
                    {
                        "name": r.get("candidate_name"),
                        "time_s": r.get("travel_time_s"),
                        "dist_m": r.get("length_m")
                    }
                    for r in (top5_abnormal[:5] if top5_abnormal else [])
                ],
                ensure_ascii=False
            ),
        })
    except Exception as e:
        print(f"[{qid}] WARNING: failed to write llm_sql_debug row: {e}")
    
    return result


def main():
    """Main routing pipeline."""
    start_ts = time.perf_counter()
    
    print("=" * 60)
    print(f"Routing Pipeline for {CITY_NAME} (Table: {TABLE_NAME})")
    print("=" * 60)
    
    # Load static info
    city_slug = CITY_SLUG
    static_info_file = STATIC_INFO_FILE
    if not static_info_file.exists():
        raise SystemExit(f"Static info file not found: {static_info_file}")
    
    print(f"Loading static info from {static_info_file.name}...")
    
    static_info_list = []
    try:
        with open(static_info_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try parsing as complete JSON array first
        try:
            static_info_list = json.loads(content)
            if not isinstance(static_info_list, list):
                static_info_list = [static_info_list]
        except json.JSONDecodeError as e:
            # JSON is malformed - try to extract valid JSON objects
            print(f"Warning: JSON parse error at position {e.pos}: {e.msg}")
            print("Attempting to extract valid JSON objects from file...")
            
            # Remove opening bracket if present
            content = content.strip()
            if content.startswith('['):
                content = content[1:].strip()
            
            # Try to add closing bracket if missing and retry
            if not content.endswith(']'):
                content_with_bracket = content.rstrip() + '\n]'
                try:
                    static_info_list = json.loads(content_with_bracket)
                    if not isinstance(static_info_list, list):
                        static_info_list = [static_info_list]
                    print(f"Successfully parsed by adding closing bracket")
                except json.JSONDecodeError:
                    # Still fails - extract objects manually
                    pass
            
            # If still empty, extract JSON objects manually
            if not static_info_list:
                # Remove closing bracket if present
                if content.endswith(']'):
                    content = content[:-1].strip()
                
                # Extract complete JSON objects by finding balanced braces
                i = 0
                while i < len(content):
                    # Skip whitespace and commas
                    while i < len(content) and content[i] in ' \n\t\r,':
                        i += 1
                    if i >= len(content):
                        break
                    
                    # Find start of JSON object
                    if content[i] == '{':
                        brace_count = 0
                        start_idx = i
                        in_string = False
                        escape_next = False
                        
                        # Find matching closing brace
                        for j in range(i, len(content)):
                            char = content[j]
                            
                            if escape_next:
                                escape_next = False
                                continue
                            
                            if char == '\\':
                                escape_next = True
                                continue
                            
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            
                            if not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        # Found complete object
                                        json_str = content[start_idx:j+1]
                                        try:
                                            obj = json.loads(json_str)
                                            static_info_list.append(obj)
                                        except json.JSONDecodeError:
                                            pass
                                        i = j + 1
                                        break
                        else:
                            # No closing brace found, skip this position
                            i += 1
                    else:
                        i += 1
                
                print(f"Extracted {len(static_info_list)} valid JSON objects")
    except Exception as e:
        raise SystemExit(f"Failed to parse static info JSON: {e}")
    
    # Optional test cap: process only the first N queries if MAX_QUERIES > 0
    limit = 0
    try:
        limit = int(MAX_QUERIES) if MAX_QUERIES is not None else 0
    except Exception:
        limit = 0
    if limit > 0:
        static_info_list = static_info_list[:limit]
    total = len(static_info_list)
    print(f"Loaded {total} queries from static info")
    # Capture IDs for visibility when starting processing
    ids_all = [str(obj.get("id")) for obj in static_info_list if isinstance(obj, dict) and obj.get("id") is not None]
    
    # Process queries (optionally capped by MAX_QUERIES)
    
    # Connect to DuckDB
    max_retries = 5
    retry_delay = 2
    con = None
    for attempt in range(max_retries):
        try:
            con = duckdb.connect(str(DB_FILE), read_only=True)
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
            break
        except Exception as e:
            if "lock" in str(e).lower() or "conflicting" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"Database locked. Retrying in {retry_delay} seconds... ({attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    raise SystemExit(f"Failed to connect after {max_retries} attempts.")
            else:
                raise
    
    if con is None:
        raise SystemExit("Failed to establish database connection.")
    
    try:
        con.execute("LOAD spatial;")
    except Exception:
        pass
    
    # Load city polygon and graphs
    print(f"\nLoading graphs for {CITY_NAME}...")
    city_polygon = load_city_polygon_from_master(CITY_NAME)
    if city_polygon is None:
        raise SystemExit(f"City polygon not found for {CITY_NAME}")
    
    import osmnx as ox  # type: ignore
    # Measure city graph load time including abnormal graph prep; print right before inference
    _t_graph_start = time.perf_counter()
    G = ox.graph_from_polygon(city_polygon, network_type="drive")
    sanitize_travel_time(G)
    G_abnormal, abn_stats = build_abnormal_graph_fema_simple(G, city_polygon, FEMA_SHP, CITY_NAME)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"FEMA abnormal: {abn_stats.get('infinite_H')} infinite edges, {abn_stats.get('scaled_else')} scaled edges")
    _graph_total_load_sec = time.perf_counter() - _t_graph_start
    print(f"City graph load time: {_graph_total_load_sec:.2f} sec")
    print("Starting inference...")
    
    # Setup output files
    output_dir = OUTPUT_DIR / CITY_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_json_file = output_dir / f"{city_slug}_routing_results.json"
    time_summary_csv = output_dir / f"{city_slug}_time_summary.csv"
    summary_csv_file = output_dir / f"{city_slug}_summary.csv"
    routes_dir = output_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    
    # Open files
    json_file = open(results_json_file, 'w', encoding='utf-8')
    json_file.write("[\n")
    
    if SAVE_LEGACY_SUMMARIES:
        time_csv_file = open(time_summary_csv, 'w', encoding='utf-8', newline='')
        time_writer = csv.DictWriter(time_csv_file, fieldnames=[
            "query_id", "t_parse_sec", "t_sql_gen_sec", "t_sql_exec_sec",
            "t_node_mapping_sec", "t_routing_normal_sec", "t_routing_abnormal_sec",
            "t_total_routing_sec", "pipeline_total_sec"
        ])
        time_writer.writeheader()
        # Do not create the other summary CSV
        summary_csv_file_handle = None
        csv_writer = _NullDictWriter()
    else:
        time_csv_file = None
        summary_csv_file_handle = None
        time_writer = _NullDictWriter()
        csv_writer = _NullDictWriter()
    
    # Excel-ready CSV for LLM/SQL details
    xl_csv_file = output_dir / f"{city_slug}_SQL_debug.csv"
    xl_file_handle = open(xl_csv_file, 'w', encoding='utf-8', newline='')
    xl_writer = csv.DictWriter(
        xl_file_handle,
        fieldnames=[
            "id", "query", "parsed_json", "ref_info_json", "sql_generated",
            "sql_status_json",
            "pred_candidate_count", "pred_candidates_json",
            "gt_sql_query", "gt_sql_count", "gt_sql_candidates_json",
            "top5_abnormal_json"
        ]
    )
    xl_writer.writeheader()
    
    # Routes collector for shapefiles
    routes_collector: Dict[str, Any] = {}
    
    # Process queries
    print(f"\n[{CITY_NAME}] Processing {total} queries...")
    print(f"[{CITY_NAME}] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    processed_count = 0
    is_first_item = True
    city_start_time = time.perf_counter()
    # Load GT SQL per-id map
    labels_map = _load_gt_sql_map(CITY_NAME)
    
    bar_len = 30
    for idx, static_info in enumerate(static_info_list):
        qid = static_info["id"]
        
        result = process_routing(static_info, G, G_abnormal, con, labels_map, json_file, time_writer, csv_writer, xl_writer, routes_collector)
        
        # Write JSON incrementally
        if not is_first_item:
            json_file.write(",\n")
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        indented = "\n".join("  " + line if line.strip() else line for line in json_str.split("\n"))
        json_file.write(indented)
        json_file.flush()
        is_first_item = False
        processed_count += 1
        
        # Progress reporting with progress bar
        current_idx = idx + 1
        progress = current_idx / total if total > 0 else 1.0
        filled = int(bar_len * progress)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"\r[{CITY_NAME}] |{bar}| {current_idx}/{total} ({100*progress:.1f}%)", end="", flush=True)
        if current_idx == total:
            print("")
    
    # Close JSON array
    json_file.write("\n]")
    json_file.close()
    if time_csv_file is not None:
        time_csv_file.close()
    if summary_csv_file_handle is not None:
        summary_csv_file_handle.close()
    xl_file_handle.close()
    
    # Save route shapefiles
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import LineString  # type: ignore
        
        def flatten_routes(routes_dict: Dict[str, Any], scenario: str) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            for rec in routes_dict.values():
                routes = rec.get(f"routes_{scenario}", []) or []
                for r in routes:
                    coords = r.get("geometry", {}).get("coordinates") if isinstance(r.get("geometry"), dict) else r.get("geometry")
                    if not coords:
                        continue
                    try:
                        geom = LineString(coords)
                    except Exception:
                        geom = None
                    rows.append({
                        "name": (r.get("to") or {}).get("name"),
                        "time_s": (r.get("metrics") or {}).get("time_ff_s"),
                        "dist_m": (r.get("metrics") or {}).get("distance_ff_m"),
                        "scenario": scenario,
                        "geometry": geom,
                    })
            return rows
        
        # Save routes to separate JSON file (like pipeline.py)
        routes_json_file = output_dir / "routes_city.json"
        with open(routes_json_file, 'w', encoding='utf-8') as f:
            json.dump(routes_collector, f, ensure_ascii=False, indent=2)
        print(f"Routes JSON saved: {routes_json_file.name} ({len(routes_collector)} queries)")
        
        # Also save shapefiles
        rows_n = flatten_routes(routes_collector, "normal")
        rows_a = flatten_routes(routes_collector, "abnormal")
        if rows_n:
            gpd.GeoDataFrame(rows_n, geometry="geometry", crs="EPSG:4326").to_file(routes_dir / "routes_normal.shp")
        if rows_a:
            gpd.GeoDataFrame(rows_a, geometry="geometry", crs="EPSG:4326").to_file(routes_dir / "routes_abnormal.shp")
        print(f"Route shapefiles saved under: {routes_dir}")
    except Exception as e:
        print(f"WARNING: route export failed: {e}")
    
    # Calculate total time
    city_total_sec = time.perf_counter() - city_start_time
    city_total_min = city_total_sec / 60.0
    
    print(f"\n[{CITY_NAME}] ✓ Completed: {processed_count} results")
    print(f"[{CITY_NAME}] Total routing time: {city_total_min:.2f} minutes ({city_total_sec:.1f} seconds)")
    if processed_count > 0:
        print(f"[{CITY_NAME}] Average time per query: {city_total_sec/processed_count:.2f} seconds")
    
    con.close()
    
    elapsed_min = (time.perf_counter() - start_ts) / 60.0
    print(f"\nTotal execution time: {elapsed_min:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()

