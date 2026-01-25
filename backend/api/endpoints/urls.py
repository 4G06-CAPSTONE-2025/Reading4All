from django.urls import path

from api.impl.GET_alt_text_history import get_history
from api.impl.POST_gen_alt_text import gen_alt_text_api
from api.impl.POST_upload import csrf, validate_image_api
from api.impl.POST_signup import signup
from api.impl.PUT_edit import edit_alt_text
from api.impl.POST_login import login
from api.impl.POST_logout import logout
from api.impl.GET_session import session_status
from ..impl.POST_send_verification import send_verification
from ..impl.POST_verify import verify

urlpatterns = [
    path("upload/", validate_image_api),
    path("generate-alt-text/", gen_alt_text_api),
    path("alt-text-history/", get_history),
    path("signup/", signup),
    path("cookie/", csrf),
    path("send-verification/", send_verification),
    path("verify/", verify),
    path("edit-alt-text/", edit_alt_text),
    path("login/", login),
    path("logout/", logout),
    path("session/", session_status)
]
