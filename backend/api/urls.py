from django.urls import path
from .POST_upload import validate_image_api, csrf

urlpatterns = [
    path("upload/", validate_image_api),
    path("cookie/", csrf)
]
