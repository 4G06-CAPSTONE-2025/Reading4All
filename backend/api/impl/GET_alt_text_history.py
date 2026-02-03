from django.http import HttpResponse, JsonResponse

from controller.backend_controller import backend_controller


# get alt text history api
def get_history(request):

    # API endpoint should only be used in get requests
    if request.method != "GET":
        response = JsonResponse(
            {"error": "Only GET Methods allowed on this endpoint"}, status=405
        )
        return response
    
    session_id = request.GET.get("session_id") # should be passed in as query parameter.

    if not session_id:
        response = JsonResponse(
            {"error": "session_id query parameter is needed"}, status=400
        )
        return response

    # calls backend_controller in order to reach service
    # session is hardcoded temporarily
    history = backend_controller.get_alt_text_history(session_id)

    # in the case where history is empty it wil still be returned as an empty list
    # ensures history is never undefined in frontend.
    response = JsonResponse(
        {"history": history},
        status=200
    )
    return response
