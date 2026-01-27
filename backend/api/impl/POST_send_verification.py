import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from databases.connect_supabase import get_supabase_client


@csrf_exempt
def send_verification(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not email.endswith("@mcmaster.ca"):
        return JsonResponse({"error": "Email must be a @mcmaster.ca address"}, status=400)

    if len(password) < 8:
        return JsonResponse({"error": "Password must be at least 8 characters"}, status=400)

    supabase = get_supabase_client()

    try:
        # This triggers Supabase to send the OTP email (with confirmations enabled)
        supabase.auth.sign_up({"email": email, "password": password})
        return JsonResponse({"ok": True, "message": "Verification code sent"}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
