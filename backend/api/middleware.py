from django.http import JsonResponse
from django.utils import timezone
from api.models import UserSession

PUBLIC_PATHS = {
    "/api/login/",
    "/api/signup/",
    "/api/send-verification/",
    "/api/verify/",
    "/api/session/",
}

class SessionAuthMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path

        # Allow public endpoints + non-API paths
        if path in PUBLIC_PATHS or not path.startswith("/api/"):
            return self.get_response(request)

        token = request.COOKIES.get("session_token")
        if not token:
            return JsonResponse({"error": "Not authenticated"}, status=401)

        s = UserSession.objects.filter(session_token=token,
                                       revoked_at__isnull=True).first()
        if not s or timezone.now() >= s.expires_at:
            return JsonResponse({"error": "Session expired"}, status=401)

        # Attach user_id for handlers to use
        request.user_id = s.user_id
        return self.get_response(request)
