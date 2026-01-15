from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from django.http import HttpResponse, JsonResponse
from PIL import Image
import io

valid_image_types = ['image/png','image/jpeg','image.jpg']
max_image_size = 10485760 # in bytes

#This function is only needed to get a cookie when running the api locally 
@ensure_csrf_cookie
def csrf(request):
    return HttpResponse(status=200)

#upload image api 
def validate_image(request):

    #API endpoint should only be used in post requests
    if request.method != 'POST':
        response = JsonResponse(
            {"error": "Only POST Methods allowed on this endpoint"},
            status = 405
        )
        return response
    
    if "image" not in request.FILES:
        response = JsonResponse(
                {"error": "MISSING_IMAGE"}, 
                status = 400
        )
        return response

    uploaded_img = request.FILES['image']

    #Must check if image meets size and type reqs
    if uploaded_img.content_type not in valid_image_types:
        response = JsonResponse(
            {
                "error": "INVALID_FILE_TYPE"
            },
            status = 400
        )
        return response
    
    if uploaded_img.size > max_image_size:
        response = JsonResponse(
            {
                "error": "FILE_SIZE_INVALID"
            },
            status = 400
        )
        return response


    try: 
        Image.open(uploaded_img).verify()


    except Exception:
        response = JsonResponse (
            {
                "error": "UNAUTHORIZED_ACCESS"
            },
            status = 400
        )
        return response
    

    return HttpResponse(status=200)