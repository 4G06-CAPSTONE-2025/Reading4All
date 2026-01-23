from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie

from controller.backend_controller import backend_controller


# This function is only needed to get a cookie when running the api locally
@ensure_csrf_cookie
def csrf(request):
    return HttpResponse(status=200)


# upload image api
def validate_image_api(request):

    # API endpoint should only be used in post requests
    if request.method != "POST":
        response = JsonResponse(
            {"error": "Only POST Methods allowed on this endpoint"}, status=405
        )
        return response

    uploaded_file = request.FILES
    message = backend_controller.validate_image(uploaded_file)

    if message != "Success":
        print("MESSAGE", message)
        response = JsonResponse({"error": message}, status=400)
        return response
    return HttpResponse(status=200)
