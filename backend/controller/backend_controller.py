import os

from services.alt_text_history import AltTextHistory
from services.gen_alt_text import GenAltText
from services.image_validation import ImageValidation
from services.edit_alt_text import edit_alt_text
from services.auth_service import AuthService

class BackendController:
    def __init__(self,
                 img_validator=None, 
                 history_info=None, 
                 gen_alt_text_for_img=None, 
                 auth_service=None,
                 edit_alt_text_service=None):

        self.image_validator = img_validator or ImageValidation()
        self.history_info = history_info or AltTextHistory()
        self.gen_alt_text_for_img = gen_alt_text_for_img or GenAltText()
        self.auth_service = auth_service or AuthService()
        self.edit_alt_text_service = edit_alt_text_service or edit_alt_text

    def validate_image(self, uploaded_img):
        return self.image_validator.validate_image(uploaded_img)

    def gen_alt_text(self, uploaded_img, session_id):
        return self.gen_alt_text_for_img.trigger_model(uploaded_img, session_id)

    def get_alt_text_history(self, session_id):
        return self.history_info.get_alt_text_history(session_id)

    def edit_alt_text(self, request):
        return self.edit_alt_text_service(request)

    def signup_user(self, email, password):
        return self.auth_service.signup(email, password)

backend_controller = BackendController()