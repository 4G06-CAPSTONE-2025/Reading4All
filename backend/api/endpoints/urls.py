from django.urls import path

from ..impl.GET_alt_text_history import get_history
from ..impl.POST_gen_alt_text import gen_alt_text_api
from ..impl.POST_upload import csrf, validate_image_api
from ..impl.POST_signup import signup

urlpatterns = [
    path("upload/", validate_image_api),
    path("generate-alt-text/", gen_alt_text_api),
    path("alt-text-history/", get_history),
    path("signup/", signup),
    path("cookie/", csrf),
]
