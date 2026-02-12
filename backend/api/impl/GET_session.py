# GET_session.py
from django.http import JsonResponse
from django.utils import timezone
from api.models import UserSession

def session_status(request):
    token = request.COOKIES.get("session_token")
    if not token:
        return JsonResponse({"authenticated": False}, status=401)

    s = UserSession.objects.filter(
        session_token=token,
        revoked_at__isnull=True
    ).first()

    if not s or timezone.now() >= s.expires_at:
        return JsonResponse({"authenticated": False}, status=401)

    return JsonResponse(
        {
            "authenticated": True,
            "user_id": s.user_id,
            "expires_at": s.expires_at.isoformat(),
        },
        status=200,
    )
