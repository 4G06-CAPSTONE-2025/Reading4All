from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from api.models import UserSession
from databases.connect_supabase import get_supabase_client


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
            # 1️⃣ Delete history for this session from Supabase
            supabase = get_supabase_client()
            supabase.table("history").delete().eq(
                "session_id",
                str(session.id)
            ).execute()

            # 2️⃣ Revoke session
            session.revoked_at = timezone.now()
            session.save(update_fields=["revoked_at"])

    resp = JsonResponse({"ok": True}, status=200)
    resp.delete_cookie("session_token")
    return resp