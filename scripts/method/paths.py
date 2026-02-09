"""
Common file paths configuration for EvacAgent.
Update paths here to change them across all scripts.
"""

from pathlib import Path

# Base directory
BASE = Path("/workspace/storage/DSPP-CH/EvacAgent")

# Data directories
FINAL_DATA = BASE / "Final-Data"
FULL_DATA = FINAL_DATA / "Full-Data"

# Result directory
RESULT_DIR = BASE / "Result"

# Common data files
TEST_QUERIES_FILE = FINAL_DATA / "test_queries.json"
EVAL_QUERY_CSV = FINAL_DATA / "eval_query.csv"
DIALOGUE_FILE = FULL_DATA / "EVClarify_dialogue.json"
TRUE_INTENT_FILE = FULL_DATA / "EVClarify_true_intent.json"
GT_POI_FILE = FULL_DATA / "EVClarify_gt_poi.json"
GT_ROUTES_FILE = FULL_DATA / "EVClarify_gt_routes.json"
GT_SQL_FILE = FULL_DATA / "EVClarify_gt_sql.json"

# Database path (DuckDB)
DB_PATH = BASE / "DB" / "DSPP_DB.duckdb"

# Calibration files
CALIB_DIR = FINAL_DATA / "calib"
MIN_PROB_VALUES_FILE = CALIB_DIR / "min_prob_values.txt"

# Scripts directory structure
SCRIPTS_DIR = BASE / "scripts"
ROUTING_MODULE_DIR = SCRIPTS_DIR / "routing_module"
ENV_FILE = SCRIPTS_DIR / ".env"

