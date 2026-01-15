from django.urls import path
from .views import validate_image, csrf

urlpatterns = [
    path("upload/", validate_image),
    path("cookie/", csrf)
]
