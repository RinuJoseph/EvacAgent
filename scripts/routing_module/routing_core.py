import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def require_packages():
    try:
        import osmnx as ox  
        import numpy as np  
        import matplotlib.pyplot as plt  
        import networkx as nx  
    except Exception as exc:
        raise SystemExit(
            "routing_core requires osmnx, numpy, matplotlib, networkx.\n"
            "Install: pip install osmnx numpy matplotlib networkx\n"
            f"Import error: {exc}"
        )
    return ox, np, plt, nx


def bbox_from_anchor(lat: float, lon: float, meters: int = 15000) -> Tuple[float, float, float, float]:
    ox, _np, _plt, _nx = require_packages()
    north, south, east, west = ox.utils_geo.bbox_from_point((lat, lon), dist=meters)
    return north, south, east, west


def induced_subgraph_bbox(G: "object", lat: float, lon: float, meters: int = 15000) -> "object":
    ox, np, _plt, nx = require_packages()
    north, south, east, west = bbox_from_anchor(lat, lon, meters)
    # Filter nodes by bbox
    node_ids = []
    xs = nx.get_node_attributes(G, "x")
    ys = nx.get_node_attributes(G, "y")
    for nid, x in xs.items():
        y = ys[nid]
        if south <= y <= north and west <= x <= east:
            node_ids.append(nid)
    H = G.subgraph(node_ids).copy()
 
    return H

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def filter_candidates_within_radius(df: pd.DataFrame, anchor_lat: float, anchor_lon: float, meters: int = 15000) -> pd.DataFrame:
    mask = df.apply(lambda r: haversine_m(anchor_lat, anchor_lon, float(r["Lat"]), float(r["Long"])) <= meters, axis=1)
    return df[mask].copy()


def map_points_to_nodes(G: "object", lats: List[float], lons: List[float]) -> List[int]:
    ox, _np, _plt, _nx = require_packages()
    return list(ox.distance.nearest_nodes(G, X=lons, Y=lats))


from typing import List, Tuple

from typing import List, Tuple
from shapely.geometry import LineString  

from typing import List, Tuple

def path_geometry(G: "object", path: List[int]) -> List[Tuple[float, float]]:
  
    if not path:
        return []

    ox, _np, _plt, _nx = require_packages()


    try:
        
        route_to_gdf = getattr(getattr(ox, "routing", None), "route_to_gdf", None)
        if route_to_gdf is None:
            route_to_gdf = ox.utils_graph.route_to_gdf
       
        try:
            gdf_nodes, gdf_edges = route_to_gdf(G, path)
        except ValueError:
            gdf_edges = route_to_gdf(G, path)

        edge_geoms = gdf_edges["geometry"]

        coords: List[Tuple[float, float]] = []

        for geom in edge_geoms:
          
            if geom is None or not hasattr(geom, "coords"):
                continue
            seg = list(geom.coords)
            if not seg:
                continue

            if not coords:
                coords.extend(seg)
            else:
              
                if coords[-1] == seg[0]:
                    coords.extend(seg[1:])
                else:
                    coords.extend(seg)

        if coords:
            return coords

    except Exception:
       
        pass

    # --- fallback: original simple behavior ---
    xs = {k: v for k, v in G.nodes(data="x")}
    ys = {k: v for k, v in G.nodes(data="y")}
    return [(xs[n], ys[n]) for n in path]

def plot_routes(
    G: "object",
    routes: List[List[int]],
    out_png: Path,
    anchor_node: int | None = None,
    candidate_nodes: List[int] | None = None,
    bgcolor: str = "white",
    road_color: str = "black",
    route_color: str = "green",
    start_color: str = "red",
    cand_color: str = "green",
) -> None:
    ox, _np, plt, nx = require_packages()
    # Base graph
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=road_color,
        bgcolor=bgcolor,
        show=False,
        close=False,
    )
    # Overlay routes
    for path in routes:
        coords = path_geometry(G, path)
        if len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        ax.plot(xs, ys, linewidth=2, color=route_color, alpha=0.9, zorder=4)

        # ox.plot_graph_route(G, path, route_color=route_color, route_linewidth=3, ax=ax, orig_dest_node_size=0, orig_dest_node_color=route_color)
    # Markers
    if anchor_node is not None:
        ax.scatter(G.nodes[anchor_node]["x"], G.nodes[anchor_node]["y"], s=40, c=start_color, zorder=5)
    if candidate_nodes:
        xs = [G.nodes[n]["x"] for n in candidate_nodes if n in G.nodes]
        ys = [G.nodes[n]["y"] for n in candidate_nodes if n in G.nodes]
        ax.scatter(xs, ys, s=30, c=cand_color, zorder=5)
    fig.set_size_inches(8, 8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_routes_json(qid: str, anchor: Dict[str, object], results: List[Dict[str, object]], out_json: Path, target_category: str | None = None) -> None:
    payload = {
        "id": qid,
        "anchor": anchor,
        "routes": results,
    }
    if target_category is not None:
        payload["target_category"] = target_category
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


