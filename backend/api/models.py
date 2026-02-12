import uuid
from datetime import timedelta
from django.db import models
from django.utils import timezone

class UserSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.CharField(max_length=36,
                               db_index=True)  # Supabase user UUID string
    session_token = models.CharField(max_length=128, unique=True,
                                     db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(db_index=True)
    revoked_at = models.DateTimeField(null=True, blank=True)

    @staticmethod
    def expiry_2h():
        return timezone.now() + timedelta(hours=2)
