import io
from PIL import Image

from services.image_validation import ImageValidation


#IMGVAL-UT 1: Test missing image in the request
def test_missing_image():
    validator = ImageValidation()

    empty_request = {}

    result = validator.validate_image(empty_request)

    assert result == "MISSING_IMAGE"


#IMGVAL-UT 2: Test invalid file type 
def test_invalid_file_type():

    validator = ImageValidation()

    text_file = io.BytesIO(b"Testing inputting invalid file type, using a txt file")
    text_file.content_type = "text/plain"
    text_file.size = 100

    uploaded_file = {"image": text_file}

    result = validator.validate_image(uploaded_file)

    assert result == "INVALID_FILE_TYPE"


#IMGVAL-UT 3: Test file size exceeding the limit
def test_file_size_invalid():

    validator = ImageValidation()

    large_image = io.BytesIO(b"fake image data here that exceeds size limit")
    large_image.content_type = "image/png"
    large_image.size = 200000000

    uploaded_file = {"image": large_image}

    result = validator.validate_image(uploaded_file)

    assert result == "FILE_SIZE_INVALID"


#IMGVAL-UT 4: Test corrupted image file
def test_corrupted_image():

    validator = ImageValidation()

    corrupted_image = io.BytesIO(b"This is not a valid image file")
    corrupted_image.content_type = "image/png"
    corrupted_image.size = 100

    uploaded_file = {"image": corrupted_image}

    result = validator.validate_image(uploaded_file)

    assert result == "UNAUTHORIZED_ACCESS_OR_CORRUPTED"


#IMGVAL-UT 5: Test valid image file
def test_valid_image_success():

    validator = ImageValidation()

    valid_image = Image.new("RGB", (100, 100), color="blue")
    valid_image_bytes = io.BytesIO()
    valid_image.save(valid_image_bytes, format="PNG")
    valid_image_bytes.seek(0)

    valid_image_bytes.content_type = "image/png"
    valid_image_bytes.size = valid_image_bytes.getbuffer().nbytes

    uploaded_file = {"image": valid_image_bytes}

    result = validator.validate_image(uploaded_file)

    assert result == "Success"
