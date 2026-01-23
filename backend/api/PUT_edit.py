from django.http import HttpResponse, JsonResponse
from controller.backend_controller import backend_controller


def edit_alt_text(request):

    if request.method != "PUT":
        return JsonResponse(
            {"error": "Only PUT Methods allowed on this endpoint"},
            status=405
        )

    message = backend_controller.edit_alt_text(request)

    if message != "Success":
        return JsonResponse({"error": message}, status=400)

    return HttpResponse(status=200)

