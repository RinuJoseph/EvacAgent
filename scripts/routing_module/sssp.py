#!/usr/bin/env python3

"""
Single-source shortest path utilities over travel_time weights.

Functions:
- nearest_node: get graph node nearest to a (lat, lon) point
- sssp_dijkstra: run Dijkstra from a source node, returning distances and paths dict
- reconstruct_path: fetch a path (list of node ids) from the paths dict
"""

from typing import Dict, Iterable, List, Tuple


def require_packages():
    try:
        import osmnx as ox  # type: ignore
        import networkx as nx  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "sssp requires osmnx and networkx.\n"
            "Install: pip install osmnx networkx\n"
            f"Import error: {exc}"
        )
    return ox, nx


def nearest_node(G: "object", lat: float, lon: float) -> int:
    ox, _nx = require_packages()
    # osmnx expects x=lon, y=lat
    return int(ox.distance.nearest_nodes(G, X=[lon], Y=[lat])[0])


def sssp_dijkstra(G: "object", source: int, weight: str = "travel_time") -> Tuple[Dict[int, float], Dict[int, list]]:
    _ox, nx = require_packages()
    distances: Dict[int, float]
    paths: Dict[int, list]
    distances, paths = nx.single_source_dijkstra(G, source=source, weight=weight)
    return distances, paths


def reconstruct_path(paths: Dict[int, list], source: int, target: int) -> List[int]:
    return list(paths.get(target, []))


