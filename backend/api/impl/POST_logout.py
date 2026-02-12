from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from api.models import UserSession


@csrf_exempt
def logout(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    token = request.COOKIES.get("session_token")
    if token:
        s = UserSession.objects.filter(session_token=token, revoked_at__isnull=True).first()
        if s:
            s.revoked_at = timezone.now()
            s.save(update_fields=["revoked_at"])

    resp = JsonResponse({"ok": True}, status=200)
    resp.delete_cookie("session_token")
    return resp
