from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from controller.backend_controller import backend_controller


# generate alt text api
@csrf_exempt
def gen_alt_text_api(request):

    # API endpoint should only be used in post requests
    if request.method != "POST":
        response = JsonResponse(
            {"error": "Only POST Methods allowed on this endpoint"}, status=405
        )
        return response

    image = request.FILES.get("image")
    
    if not image: 
        response = JsonResponse(
            {"error": "image is required"}, status=400
        )
        return response

    session_id = request.session_id

    print("here is session_id in gen alt text api", session_id)

    alt_text = backend_controller.gen_alt_text(image, session_id)

    if alt_text:
        response = JsonResponse({"alt_text": alt_text}, status=200)
        return response

    response = JsonResponse({"error": "ERROR_GENERATING"}, status=400)
    return response
