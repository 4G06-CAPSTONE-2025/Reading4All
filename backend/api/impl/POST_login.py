import json
import secrets
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from databases.connect_supabase import get_supabase_admin_client
from api.models import UserSession


@csrf_exempt
def login(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not email or not password:
        return JsonResponse({"error": "Email and password are required"}, status=400)

    try:
        supabase = get_supabase_admin_client()
        resp = supabase.auth.sign_in_with_password({"email": email, "password": password})

        user = getattr(resp, "user", None) or (resp.get("user") if isinstance(resp, dict) else None)
        if not user:
            return JsonResponse({"error": "Invalid credentials"}, status=401)

        user_id = getattr(user, "id", None)

        # Create YOUR app session token (2 hours)
        session_token = secrets.token_urlsafe(32)
        expires_at = UserSession.expiry_2h()

        UserSession.objects.create(
            user_id=user_id,
            session_token=session_token,
            expires_at=expires_at,
        )

        # Set cookie (frontend-friendly)
        resp_json = JsonResponse({"ok": True, "user_id": user_id, "expires_at": expires_at.isoformat()}, status=200)

        # secure=False for local dev (http). Set True in production (https).
        resp_json.set_cookie(
            "session_token",
            session_token,
            httponly=True,
            samesite="Lax",
            secure=False,
            max_age=2 * 60 * 60,
        )
        return resp_json

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
