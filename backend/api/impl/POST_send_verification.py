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

    if not email.endswith("@mcmaster.ca"):
        return JsonResponse(
            {"error": "Email must be a @mcmaster.ca address"},
            status=400
        )

    supabase = get_supabase_client()

    # 1️⃣ Check if user already exists
    try:
        existing_user = supabase.auth.admin.get_user_by_email(email)
        if existing_user and existing_user.user:
            return JsonResponse(
                {"error": "User already registered. Please log in."},
                status=400
            )
    except Exception:
        # ✅ Exception means user does NOT exist
        pass

    # 2️⃣ Send OTP for new user
    try:
        supabase.auth.sign_in_with_otp({
            "email": email,
            "options": {
                "should_create_user": True
            }
        })

        return JsonResponse(
            {"ok": True, "message": "OTP code sent"},
            status=200
        )

    except Exception as e:
        # Optional: log e for debugging
        return JsonResponse(
            {"error": "Failed to send verification code"},
            status=400
        )