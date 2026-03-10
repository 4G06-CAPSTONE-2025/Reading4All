from databases.connect_supabase import get_supabase_client


class AuthService:
    def __init__(self, supabase=None):
        self.supabase = supabase or get_supabase_client()

    def signup(self, email: str, password: str):
        email = (email or "").strip().lower()

        # Removed domain restriction ✅

        if len(password or "") < 8:
            raise ValueError("Password must be at least 8 characters")

        # Supabase Auth signup
        resp = supabase.auth.sign_up({
            "email": email,
            "password": password
        })

        # supabase-py return formats can differ; support both
        user = getattr(resp, "user", None) or (
            resp.get("user") if isinstance(resp, dict) else None
        )
        session = getattr(resp, "session", None) or (
            resp.get("session") if isinstance(resp, dict) else None
        )

        user_id = getattr(user, "id", None) if user else None
        access_token = getattr(session, "access_token", None) if session else None

        return {
            "user_id": user_id,
            "access_token": access_token
        }