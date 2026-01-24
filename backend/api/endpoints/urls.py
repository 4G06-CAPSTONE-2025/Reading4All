from django.urls import path

from api.impl.GET_alt_text_history import get_history
from api.impl.POST_gen_alt_text import gen_alt_text_api
from api.impl.POST_upload import csrf, validate_image_api
from api.impl.POST_signup import signup
from api.impl.PUT_edit import edit_alt_text


urlpatterns = [
    path("upload/", validate_image_api),
    path("generate-alt-text/", gen_alt_text_api),
    path("alt-text-history/", get_history),
    path("signup/", signup),
    path("cookie/", csrf),
    path("edit-alt-text/", edit_alt_text)
]
