from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from django.http import HttpResponse, JsonResponse

from backend.backend_controller import backend_controller


# generate alt text api
def gen_alt_text_api(request):

    # API endpoint should only be used in post requests
    if request.method != "POST":
        response = JsonResponse(
            {"error": "Only POST Methods allowed on this endpoint"}, status=405
        )
        return response

    image = request.FILES["image"]

    # mocking getting a session_id from the table
    session_id = 2026
    alt_text = backend_controller.gen_alt_text(image, session_id)

    if alt_text:
        response = JsonResponse({"alt_text": alt_text}, status=200)
        return response

    else:
        response = JsonResponse({"error": "ERROR_GENERATING"}, status=400)
        return response
