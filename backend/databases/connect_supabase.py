from pathlib import Path
import yaml
from supabase import create_client

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
# __file__ = backend/databases/connect_supabase.py
# parents[1] = backend/

with CONFIG_PATH.open(encoding="utf-8") as f:
    config = yaml.safe_load(f)

SUPABASE_URL = config["PROJECT_URL"]
SUPABASE_KEY = config["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
