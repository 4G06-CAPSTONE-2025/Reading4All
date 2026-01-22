from django.urls import path

from .GET_alt_text_history import get_history
from .POST_gen_alt_text import gen_alt_text_api
from .POST_upload import csrf, validate_image_api

urlpatterns = [
    path("upload/", validate_image_api),
    path("generate-alt-text/", gen_alt_text_api),
    path("alt-text-history/", get_history),
    path("cookie/", csrf),
]
