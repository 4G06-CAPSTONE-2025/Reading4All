import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load backend/.env no matter how this module is imported
BACKEND_DIR = Path(__file__).resolve().parents[1]  # .../backend/
load_dotenv(BACKEND_DIR / ".env")


def get_supabase_client():
    """
    Public / user-scoped Supabase client.
    Uses ANON key.
    """
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not anon_key:
        raise EnvironmentError(
            "Missing Supabase ANON credentials. "
            "Expected SUPABASE_URL and SUPABASE_ANON_KEY in backend/.env"
        )

    return create_client(url, anon_key)


def get_supabase_admin_client():
    """
    Admin Supabase client.
    Uses SERVICE ROLE key.
    Backend-only. Never expose this to frontend.
    """
    url = os.getenv("SUPABASE_URL")
    role_key = os.getenv("SUPABASE_ROLE_KEY")

    if not url or not role_key:
        raise EnvironmentError(
            "Missing Supabase SERVICE ROLE credentials. "
            "Expected SUPABASE_URL and SUPABASE_ROLE_KEY in backend/.env"
        )

    return create_client(url, role_key)
