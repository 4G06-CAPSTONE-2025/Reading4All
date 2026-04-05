from services.edit_alt_text import edit_alt_text
from unittest.mock import MagicMock, patch
import json

@patch("services.edit_alt_text.AltTextHistory")
def test_edit_alt_text_success(mock_history_class):
    mock_history = MagicMock()
    mock_history.update_edited_alt_text.return_value = True 
    mock_history_class.return_value = mock_history

    request = MagicMock()
    request.body = json.dumps(
        {
            "entry_id": "101",
            "edited_alt_text":"new changed text"
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request)

    mock_history.update_edited_alt_text.assert_called_once_with(
        session_id = "1",
        entry_id="101",
        edited_alt_text="new changed text"
    )

    assert result == "Success"


@patch("services.edit_alt_text.AltTextHistory")
def test_edit_alt_text_unable_to_save(mock_history_class):
    mock_history = MagicMock()
    mock_history.update_edited_alt_text.return_value = False 
    mock_history_class.return_value = mock_history

    request = MagicMock()
    request.body = json.dumps(
        {
            "entry_id": "101",
            "edited_alt_text":"new changed text"
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request)

    mock_history.update_edited_alt_text.assert_called_once_with(
        session_id = "1",
        entry_id="101",
        edited_alt_text="new changed text"
    )

    assert result == "UNABLE_TO_SAVE"


@patch("services.edit_alt_text.AltTextHistory")
def test_edit_alt_text_invalid_missing_entry_id(mock_history_class):
    request = MagicMock()
    request.body = json.dumps(
        {
            "edited_alt_text":"new changed text"
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request)
    assert result == "INVALID_REQUEST"
    mock_history_class.return_value.update_edited_alt_text.assert_not_called()


@patch("services.edit_alt_text.AltTextHistory")
def test_edit_alt_text_invalid_missing_alt_text(mock_history_class):
   
    request = MagicMock()
    request.body = json.dumps(
        {
            "entry_id": "101",
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request)
    assert result == "INVALID_REQUEST"
    mock_history_class.return_value.update_edited_alt_text.assert_not_called()


@patch("services.edit_alt_text.AltTextHistory")
def test_edit_alt_text_exception_thrown(mock_history_class):
    mock_history = MagicMock()
    mock_history.update_edited_alt_text.side_effect = Exception("SUPABASE ERROR") 
    mock_history_class.return_value= mock_history

    request = MagicMock()
    request.body = json.dumps(
        {
            "edited_alt_text":"new changed text",
            "entry_id": "101",
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request)
    assert result == "UNABLE_TO_SAVE"
