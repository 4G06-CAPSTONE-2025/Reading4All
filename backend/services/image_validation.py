from PIL import Image

class ImageValidation: 

    def __init__(self):
        self.img_types = ['image/png','image/jpeg','image.jpg']
        self.max_img_size = 10485760 #equivalent to 10 MB

    def validate_image(self,uploaded_file):
        
        # Must check if request is missing an image
        if "image" not in uploaded_file:
            return "MISSING_IMAGE"
        

        uploaded_img = uploaded_file['image']

        # Must check if image meets type requirements 
        if uploaded_img.content_type not in self.img_types:
            return "INVALID_FILE_TYPE"
        
        # Must check if image meets size requirements
        if uploaded_img.size > self.max_img_size:
            return "FILE_SIZE_INVALID"
        
        # Must check if image can be opened, otherwise permission issue or corrupted
        try: 
            Image.open(uploaded_img).verify()


        except Exception:
            return "UNAUTHORIZED_ACCESS_OR_CORRUPTED"
    
        return "Success"