from databases.connect_supabase import get_supabase_client

class AuthService:
    def signup(self, email: str, password: str):
        email = (email or "").strip().lower()

        if not email.endswith("@mcmaster.ca"):
            raise ValueError("Email must be a @mcmaster.ca address")

        if len(password or "") < 8:
            raise ValueError("Password must be at least 8 characters")

        supabase = get_supabase_client()

        # Supabase Auth signup
        resp = supabase.auth.sign_up({"email": email, "password": password})

        # supabase-py return formats can differ; support both
        user = getattr(resp, "user", None) or (resp.get("user") if isinstance(resp, dict) else None)
        session = getattr(resp, "session", None) or (resp.get("session") if isinstance(resp, dict) else None)

        user_id = getattr(user, "id", None) if user else None
        access_token = getattr(session, "access_token", None) if session else None

        return {"user_id": user_id, "access_token": access_token}