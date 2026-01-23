import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from databases.connect_supabase import get_supabase_client


@csrf_exempt
def verify(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    email = (body.get("email") or "").strip().lower()
    token = (body.get("token") or "").strip()

    if not email:
        return JsonResponse({"error": "Email is required"}, status=400)
    if not token:
        return JsonResponse({"error": "Verification code is required"}, status=400)

    supabase = get_supabase_client()

    try:
        # Email OTP verification
        resp = supabase.auth.verify_otp(
            {"email": email, "token": token, "type": "email"}
        )

        # handle object/dict response styles
        user = getattr(resp, "user", None) or (resp.get("user") if isinstance(resp, dict) else None)
        session = getattr(resp, "session", None) or (resp.get("session") if isinstance(resp, dict) else None)

        user_id = getattr(user, "id", None) if user else None
        access_token = getattr(session, "access_token", None) if session else None
        refresh_token = getattr(session, "refresh_token", None) if session else None

        return JsonResponse(
            {
                "ok": True,
                "user_id": user_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
            },
            status=200,
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
