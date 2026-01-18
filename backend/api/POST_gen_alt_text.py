from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from django.http import HttpResponse, JsonResponse

from backend_controller import backend_controller


#generate alt text api 
def gen_alt_text_api(request):

    #API endpoint should only be used in post requests
    if request.method != 'POST':
        response = JsonResponse(
            {"error": "Only POST Methods allowed on this endpoint"},
            status = 405
        )
        return response
    

    
    image = request.FILES['image']
    message = backend_controller.gen_alt_text(image)

    if message:
        return HttpResponse(status=200)
    else:
        response = JsonResponse(
            {
                "error": "ERROR_GENERATING"
            },
            status = 400
        )
        return response