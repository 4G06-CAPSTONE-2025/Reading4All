from django.http import JsonResponse

from controller.backend_controller import backend_controller


# generate alt text api
def gen_alt_text_api(request):

    # API endpoint should only be used in post requests
    if request.method != "POST":
        response = JsonResponse(
            {"error": "Only POST Methods allowed on this endpoint"}, status=405
        )
        return response

    image = request.FILES.get("image")
    session_id = request.POST.get("session_id") # should also be passed in as form-data
    
    if not image: 
        response = JsonResponse(
            {"error": "image is required"}, status=400
        )
        return response

    if not session_id:
        response = JsonResponse(
            {"error": "session_id is required"}, status=400
        )
        return response

    alt_text = backend_controller.gen_alt_text(image, session_id)

    if alt_text:
        response = JsonResponse({"alt_text": alt_text}, status=200)
        return response

    response = JsonResponse({"error": "ERROR_GENERATING"}, status=400)
    return response
