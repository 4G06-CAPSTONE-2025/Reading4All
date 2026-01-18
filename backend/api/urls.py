from django.urls import path
from .POST_upload import validate_image_api, csrf
from .POST_gen_alt_text import gen_alt_text_api
from .GET_alt_text_history import get_history

urlpatterns = [
    path("upload/", validate_image_api),
    path("generate-alt-text/",gen_alt_text_api),
    path("alt-text-history/", get_history),
    path("cookie/", csrf)
]
