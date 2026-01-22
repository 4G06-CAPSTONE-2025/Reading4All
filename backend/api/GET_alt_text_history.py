from django.http import HttpResponse, JsonResponse

from backend.backend_controller import backend_controller


# get alt text history api
def get_history(request):

    # API endpoint should only be used in get requests
    if request.method != "GET":
        response = JsonResponse(
            {"error": "Only GET Methods allowed on this endpoint"}, status=405
        )
        return response

    # calls backend_controller in order to reach service
    # session is hardcoded temporarily
    if backend_controller.get_alt_text_history(session_id=2026):
        return HttpResponse(status=200)
    response = JsonResponse({"error": "NO_HISTORY_FOUND"}, status=400)
    return response
