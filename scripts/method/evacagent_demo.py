
"""
Demo script for EvacAgent pipeline.
Runs the entire pipeline on a single query for demonstration purposes.
Can accept either a query_id (from EVClarify) or a query string directly.
"""

from __future__ import annotations

import json
import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

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

# Import functions from eval.py
from eval import (
    get_client_for_model,
    get_model_type,
    parse_query,
    load_min_prob_values,
    run_one_query,
    TOP_K,
)

# Use paths from paths.py
BASE = paths.BASE
FINAL_DATA = paths.FINAL_DATA
FULL_DATA = paths.FULL_DATA
DIALOGUE_FILE = paths.DIALOGUE_FILE
TRUE_INTENT_FILE = paths.TRUE_INTENT_FILE
ENV_FILE = paths.ENV_FILE

# Load environment variables
load_dotenv(str(ENV_FILE))


def load_query_by_id(query_id: str) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load query data from EVClarify files by query_id.
    
    Returns:
        tuple: (record, dialogue_record)
    """
    # Load dialogue data first (contains query text)
    dialogue_data = {}
    dialogue_file = Path(DIALOGUE_FILE)
    if dialogue_file.exists():
        with open(dialogue_file) as f:
            dialogue_list = json.load(f)
            dialogue_data = {rec["query_id"]: rec for rec in dialogue_list}
    
    # Load true intent data
    true_intent_data = {}
    true_intent_file = Path(TRUE_INTENT_FILE)
    if true_intent_file.exists():
        with open(true_intent_file) as f:
            true_intent_list = json.load(f)
            true_intent_data = {rec["query_id"]: rec for rec in true_intent_list}
    
    # Get data for this query_id
    dialogue_rec = dialogue_data.get(query_id)
    true_intent_rec = true_intent_data.get(query_id)
    
    if not true_intent_rec and not dialogue_rec:
        raise SystemExit(f"Query ID '{query_id}' not found in EVClarify files")
    
    # Extract query from dialogue record if available
    query = ""
    if dialogue_rec and dialogue_rec.get("query"):
        query = dialogue_rec.get("query")
    elif not query:
        # Try to get from eval_query.csv as fallback
        eval_csv = FINAL_DATA / "eval_query.csv"
        if eval_csv.exists():
            import csv
            with open(eval_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["query_id"] == query_id:
                        query = row["query"]
                        break
    
    if not query:
        raise SystemExit(f"Could not find query text for query_id '{query_id}'")
    
    # Get city and true_intent from true_intent_rec if available, otherwise from dialogue_rec
    city = "Chicago"
    true_intent = {}
    
    if true_intent_rec:
        city = true_intent_rec.get("city", "Chicago")
        true_intent = true_intent_rec.get("true_intent", {})
    elif dialogue_rec:
        # Try to get from dialogue record
        if dialogue_rec.get("true_intent"):
            true_intent = dialogue_rec.get("true_intent", {})
    
    # Create record
    record = {
        "query_id": query_id,
        "query": query,
        "city": city,
        "true_intent": true_intent,
    }
    
    return record, dialogue_rec


def print_result_summary(result: Dict[str, Any]) -> None:
    """Print a simplified summary of the result."""
    query_id = result.get("query_id", "N/A")
    query = result.get("query", "N/A")
    
    print(f"\nQuery ID: {query_id}")
    print(f"Query: {query}")
    
    # Dialogue turns
    dialogue = result.get("dialogue", [])
    print(f"\nDialogue turns: {len(dialogue)}")
    if dialogue:
        for turn in dialogue:
            turn_num = turn.get("turn", 0)
            qtype = turn.get("question_type", "Unknown")
            question = turn.get("question_text", f"Which {qtype.lower().replace('ask', '')}?")
            answer = turn.get("answer", "N/A")
            print(f"  Turn {turn_num}: {qtype} -> {answer}")
    else:
        print("  (No dialogue turns)")
    
    # SQL
    sql = result.get("generated_sql")
    if sql:
        print(f"\nSQL: {sql}")
    else:
        print("\nSQL: Not generated")
    
    # Candidate count
    num_candidates = result.get("num_candidates", 0)
    print(f"\nCandidate count: {num_candidates}")
    
    # Final routing POIs
    final_poi = result.get("final_poi", [])
    print(f"\nFinal routing POIs: {len(final_poi)}")
    if final_poi:
        for idx, poi in enumerate(final_poi, 1):
            name = poi.get("poi_name", "N/A")
            ptype = poi.get("poi_type", "N/A")
            cost = poi.get("route_cost_s", 0)
            dist = poi.get("distance_m", 0)
            print(f"  {idx}. {name} ({ptype}) - cost: {cost:.1f}s, dist: {dist:.1f}m")
    else:
        print("  (No routing POIs)")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    ap = argparse.ArgumentParser(
        description="EvacAgent Demo - Run pipeline on a single query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with query_id from EVClarify
  python evacagent_demo.py --query_id EA-000
  
  # Run with custom query string
  python evacagent_demo.py --query "From Ventra, find emergency services within 5 km" --city Chicago
        """
    )
    
    ap.add_argument("--model", type=str, default="gpt-4o-mini",
                    help="Model name (default: gpt-4o-mini)")
    ap.add_argument("--query_id", type=str, default=None,
                    help="Query ID from EVClarify (e.g., EA-000)")
    ap.add_argument("--query", type=str, default=None,
                    help="Query string (if not using query_id)")
    ap.add_argument("--city", type=str, default="Chicago",
                    help="City name (only used if --query is provided, default: Chicago)")
    ap.add_argument("--max_turns", type=int, default=5,
                    help="Maximum dialogue turns (default: 5)")
    ap.add_argument("--scenario", type=str, default="abnormal",
                    help="Scenario type (default: abnormal)")
    ap.add_argument("--save_result", action="store_true", default=False,
                    help="Save simplified result to JSON (only printed fields + routes)")
    
    args = ap.parse_args()
    
    # Validate arguments
    if not args.query_id and not args.query:
        ap.error("Must provide either --query_id or --query")
    if args.query_id and args.query:
        ap.error("Cannot provide both --query_id and --query (use one or the other)")
    
    # Determine model type
    model_type = get_model_type(args.model)
    
    # Load min_prob values
    target_min_prob, mode_min_prob = load_min_prob_values()
    
    # Get client
    try:
        client = get_client_for_model(args.model)
    except Exception as e:
        raise SystemExit(f"Failed to initialize client for model '{args.model}': {e}")
    
    # Load query data
    if args.query_id:
        record, dialogue_record = load_query_by_id(args.query_id)
        query_id = args.query_id
        query = record["query"]
        city = record.get("city", "Chicago")
    else:
        query_id = f"DEMO-{args.query[:20].replace(' ', '_')}"
        query = args.query
        city = args.city
        record = {
            "query_id": query_id,
            "query": query,
            "city": city,
            "true_intent": {},  # No GT available for custom queries
        }
        dialogue_record = None
    
    # Run the pipeline (suppress verbose output)
    import io
    import contextlib
    
    # Suppress print statements from run_one_query
    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            result = run_one_query(
                client,
                query_id,
                query,
                record,
                dialogue_record,
                args.model,
                model_type,
                args.max_turns,
                args.scenario,
                target_min_prob,
                mode_min_prob,
                city=city,
            )
        # Generate route geometry for final_poi if available
        routes_abnormal = []
        if result.get("final_poi") and result.get("final_intent") and result.get("candidate_list"):
            final_intent = result.get("final_intent", {})
            anchor = final_intent.get("anchor", {})
            anchor_lat = anchor.get("lat")
            anchor_lon = anchor.get("lon")
            mode = final_intent.get("mode", "drive")
            
            if anchor_lat and anchor_lon:
                try:
                    import osmnx as ox
                    from routing_module.routing_pipeline import build_route_geometry_lonlat
                    from routing_module.routing_core import map_points_to_nodes
                    
                    GRAPH_CACHE_DIR = BASE / "Data" / "graph-cache"
                    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    mode_label = mode.lower() if isinstance(mode, str) else "drive"
                    
                    # Get city name for graph file (convert spaces to underscores for filename)
                    city_name = record.get("city", "Chicago")
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
                            print(f"Warning: {city_name_for_graph} graph not found, using Chicago graph as fallback")
                    
                    if graph_path.exists():
                        G = ox.load_graphml(graph_path)
                        
                        # Build mapping from POI name to coordinates from candidate_list
                        poi_name_to_coords = {}
                        for cand in result.get("candidate_list", []):
                            name = cand.get("name")
                            if name and cand.get("lat") and cand.get("lon"):
                                poi_name_to_coords[name] = (cand.get("lat"), cand.get("lon"))
                        
                        # Get POI coordinates for final_poi
                        poi_coords = []
                        for poi_result in result.get("final_poi", []):
                            poi_name = poi_result.get("poi_name")
                            if poi_name in poi_name_to_coords:
                                poi_coords.append(poi_name_to_coords[poi_name])
                        
                        if poi_coords:
                            all_lats = [anchor_lat] + [lat for lat, lon in poi_coords]
                            all_lons = [anchor_lon] + [lon for lat, lon in poi_coords]
                            all_nodes = map_points_to_nodes(G, all_lats, all_lons)
                            
                            if all_nodes and all_nodes[0] is not None:
                                src_node = all_nodes[0]
                                
                                # Compute routes for each POI
                                from routing_module.sssp import sssp_dijkstra, reconstruct_path
                                distances, route_paths = sssp_dijkstra(G, src_node, weight_key)
                                
                                for idx, (poi_lat, poi_lon) in enumerate(poi_coords):
                                    if idx + 1 < len(all_nodes) and all_nodes[idx + 1] is not None:
                                        tgt_node = all_nodes[idx + 1]
                                        if tgt_node in distances:
                                            path = reconstruct_path(route_paths, src_node, tgt_node)
                                            if path:
                                                geom_coords, _ = build_route_geometry_lonlat(G, path, weight_key=weight_key, length_key="length")
                                                poi_result = result.get("final_poi", [])[idx]
                                                routes_abnormal.append({
                                                    "name": poi_result.get("poi_name", ""),
                                                    "fclass": poi_result.get("poi_type", ""),
                                                    "lat": poi_lat,
                                                    "long": poi_lon,
                                                    "polyline_lonlat": geom_coords,
                                                })
                except Exception as e:
                    print(f"Warning: Could not generate route geometry: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Add routes to result
        result["routes_abnormal"] = routes_abnormal
        
        # Print summary
        print_result_summary(result)
        
        # Save simplified result if --save_result flag is set
        if args.save_result:
            simplified_result = {
                "query_id": result.get("query_id"),
                "query": result.get("query"),
                "dialogue": result.get("dialogue", []),
                "generated_sql": result.get("generated_sql"),
                "num_candidates": result.get("num_candidates", 0),
                "final_poi": result.get("final_poi", []),
                "routes_abnormal": routes_abnormal,
            }
            
            output_file = paths.RESULT_DIR / "demo_result.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_result, f, indent=2, ensure_ascii=False)
            print(f"\nResult saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

