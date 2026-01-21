from supabase import create_client
import yaml

with open("backend/config.yaml") as f:
    config = yaml.safe_load(f)
SUPABASE_URL = config["PROJECT_URL"]
SUPABASE_KEY = config["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)