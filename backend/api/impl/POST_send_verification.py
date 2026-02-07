import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from databases.connect_supabase import (
    get_supabase_client,
    get_supabase_admin_client,
)


@csrf_exempt
def send_verification(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    email = (body.get("email") or "").strip().lower()

    if not email.endswith("@mcmaster.ca"):
        return JsonResponse(
            {"error": "Email must be a @mcmaster.ca address"},
            status=400,
        )

    admin = get_supabase_admin_client()
    public = get_supabase_client()

    # 1️⃣ Check if user already exists (ADMIN)
    try:
        admin.auth.admin.get_user_by_email(email)
        return JsonResponse(
            {"error": "User already registered. Please log in."},
            status=400,
        )
    except Exception:
        # user does not exist → expected
        pass

    # 2️⃣ Send OTP (PUBLIC)
    try:
        public.auth.sign_in_with_otp({
            "email": email,
            "options": {"should_create_user": True},
        })

        return JsonResponse(
            {"ok": True, "message": "OTP code sent"},
            status=200,
        )

    except Exception as e:
        print("OTP SEND ERROR:", e)
        return JsonResponse(
            {"error": str(e)},
            status=400,
        )