from pathlib import Path
from typing import Tuple


def require_packages():
    try:
        import geopandas as gpd  
        import osmnx as ox  
    except Exception as exc:
        raise SystemExit(
            "graph_loader requires geopandas and osmnx.\n"
            "Install: pip install geopandas osmnx\n"
            f"Import error: {exc}"
        )
    return gpd, ox


BASE_DIR = Path("/workspace/storage/L2OPT-Data-final/Disaster-Path-Planning")
SHP_DIR = BASE_DIR / "shp/chicago-poly"


def load_chicago_polygon() -> "object":
    gpd, _ox = require_packages()
    shp_files = sorted(SHP_DIR.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp in {SHP_DIR}")
    gdf = gpd.read_file(shp_files[0])
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)
    geom = gdf.geometry.unary_union
    return geom


def load_drive_graph_freeflow() -> Tuple["object", "object"]:
    """Return (G, polygon) for Chicago drive network with travel_time seconds."""
    gpd, ox = require_packages()
    chicago_poly = load_chicago_polygon()
    G = ox.graph_from_polygon(
        chicago_poly,
        network_type="drive",
        simplify=True,
        retain_all=False,
        clean_periphery=True,
    )
    # Add speeds and travel times (free-flow defaults by highway type)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G, chicago_poly





