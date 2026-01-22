from services.alt_text_history import AltTextHistory
from services.gen_alt_text import GenAltText
from services.image_validation import ImageValidation


class BackendController:
    def __init__(self):
        self.image_validator = ImageValidation()
        self.history_info = AltTextHistory()
        self.gen_alt_text_for_img = GenAltText()

    def validate_image(self, uploaded_img):
        return self.image_validator.validate_image(uploaded_img)

    def gen_alt_text(self, uploaded_img, session_id):
        return self.gen_alt_text_for_img.trigger_model(uploaded_img, session_id)

    def get_alt_text_history(self, session_id):
        return self.history_info.get_alt_text_history(session_id)


backend_controller = BackendController()
