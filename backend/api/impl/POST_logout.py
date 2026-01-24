from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from api.models import UserSession, History


@csrf_exempt
def logout(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    token = request.COOKIES.get("session_token")

    if token:
        session = UserSession.objects.filter(
            session_token=token,
            revoked_at__isnull=True
        ).first()

        if session:
            # delete all history rows for this session
            History.objects.filter(session_id=session.id).delete()

            # revoke the session
            session.revoked_at = timezone.now()
            session.save(update_fields=["revoked_at"])

    resp = JsonResponse({"ok": True}, status=200)
    resp.delete_cookie("session_token")
    return resp