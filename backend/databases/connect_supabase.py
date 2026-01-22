import os
from supabase import create_client

supabase = None

def get_supabase_client():
    global supabase

    if supabase is None:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not SUPABASE_URL or not SUPABASE_KEY:
            raise EnvironmentError("Supabase URL or Key not found in environment variables.")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


    return supabase