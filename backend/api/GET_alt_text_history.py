from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from django.http import HttpResponse, JsonResponse

from backend_controller import backend_controller

#generate alt text api 
def get_history(request):

    #API endpoint should only be used in get requests
    if request.method != 'GET':
        response = JsonResponse(
            {"error": "Only GET Methods allowed on this endpoint"},
            status = 405
        )
        return response
    if backend_controller.get_alt_text_history(session_id=2026):
         return HttpResponse(status=200)
    else:
        response = JsonResponse(
            {
                "error": "NO_HISTORY_FOUND"
            },
            status = 400
        )
        return response



    