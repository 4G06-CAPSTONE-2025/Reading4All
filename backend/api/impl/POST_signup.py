import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from controller.backend_controller import backend_controller


@csrf_exempt
def signup(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    email = (body.get("email") or "").strip()
    password = body.get("password") or ""

    try:
        result = backend_controller.signup_user(email, password)
        return JsonResponse({"ok": True, **result}, status=201)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
