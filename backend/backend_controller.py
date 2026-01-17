
from services.image_validation import ImageValidation
class BackendController: 
    def __init__(self):
        self.image_validator = ImageValidation()
    
    def validate_image(self,uploaded_img):
        return self.image_validator.validate_image(uploaded_img)

backend_controller = BackendController()
