"""
Author: Moly Mikhail 
Date: March 2026
Purpose: Tests for the BackendController module, 
specifically ensuring the routing to other modules is done
correctly.
"""

from unittest.mock import MagicMock, patch

# mocking all the external services that 
# backend controller routes data/requests to 
with (
    patch("services.alt_text_history.AltTextHistory", MagicMock()),
    patch("services.gen_alt_text.GenAltText", MagicMock()),
    patch("services.image_validation.ImageValidation", MagicMock()),
    patch("services.edit_alt_text.edit_alt_text", MagicMock()),
    patch("services.auth_service.AuthService", MagicMock()),
):
    from controller.backend_controller import BackendController


# BCONT-UT1: Tests that the backend controller correctly
# routes image validation requests to the Image Validation module.
def test_validate_image_routing():
    controller = BackendController()

    valid_img_return_value = "Success"
    controller.image_validator.validate_image = MagicMock(
        return_value=valid_img_return_value
        )
    result = controller.validate_image("Fake image data")

    controller.image_validator.validate_image.assert_called_once_with("Fake image data")
    assert result == "Success"

# BCONT-UT2: Tests that the backend controller correctly routes alt text generation
# requests to the Gen Alt Text module and returns the generated alt text.
def test_gen_alt_text_routing():

    controller = BackendController()

    gen_alt_text_return = "Generated alt text"
    controller.gen_alt_text_for_img.trigger_model = MagicMock(
        return_value=gen_alt_text_return
        )
    result = controller.gen_alt_text("Fake image data", "fake_session_id")

    controller.gen_alt_text_for_img.trigger_model.assert_called_once_with(
        "Fake image data", "fake_session_id"
        )
    assert result == "Generated alt text"

# BCONT-UT3: Tests that the backend controller correctly routes alt text history
# retrieval requests to the Alt Text History module
# and returns the retrieved history.
def test_get_alt_text_history_routing():

    controller = BackendController()

    get_alt_history_return = ["Alt text 1", "Alt text 2"]
    controller.history_info.get_alt_text_history = MagicMock(
        return_value=get_alt_history_return
        )
    result = controller.get_alt_text_history(
        "fake_session_id"
        )

    controller.history_info.get_alt_text_history.assert_called_once_with(
        "fake_session_id"
        )
    assert result == ["Alt text 1", "Alt text 2"]

# BCONT-UT4: Tests that the backend controller correctly routes
# alt text editing requests to the Edit Alt Text module
# and returns the editing results.
def test_edit_alt_text_routing():

    controller = BackendController()

    edit_alt_text_return = "Success"
    controller.edit_alt_text_service = MagicMock(
        return_value=edit_alt_text_return
        )

    result = controller.edit_alt_text("mock_request")

    controller.edit_alt_text_service.assert_called_once_with("mock_request")
    assert result == "Success"
