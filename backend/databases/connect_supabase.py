import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load backend/.env no matter how this module is imported
BACKEND_DIR = Path(__file__).resolve().parents[1]  # .../backend/
load_dotenv(BACKEND_DIR / ".env")

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")  # matches your env var names

    if not url or not key:
        raise EnvironmentError(
            "Supabase URL or Key not found in environment variables. "
            "Expected SUPABASE_URL and SUPABASE_KEY in backend/.env"
        )

    return create_client(url, key)
