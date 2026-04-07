"""
Author: Moly Mikhail
Date: March 2026
Purpose: Unit tests for the editing alt text,
specifically verifying valid and invalid edit requests handling. 
"""

from unittest.mock import MagicMock
import json
from services.edit_alt_text import edit_alt_text

# EDIT-UT1: Tests that a valid edit request is routed correctly and returns success
def test_edit_alt_text_success():

    # mock AltTextHistory client as its not being tested
    mock_history = MagicMock()
    mock_history.update_edited_alt_text.return_value = True

    request = MagicMock()
    request.body = json.dumps(
        {
            "entry_id": "101",
            "edited_alt_text":"new changed text"
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request, history=mock_history)

    mock_history.update_edited_alt_text.assert_called_once_with(
        session_id = "1",
        entry_id="101",
        edited_alt_text="new changed text"
    )

    assert result == "Success"

# EDIT-UT2: Tests that failing to save to DB returns the appropriate error
def test_edit_alt_text_unable_to_save():
    # mock AltTextHistory client as its not being tested
    mock_history = MagicMock()
    mock_history.update_edited_alt_text.return_value = False

    request = MagicMock()
    request.body = json.dumps(
        {
            "entry_id": "101",
            "edited_alt_text":"new changed text"
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request, history=mock_history)

    mock_history.update_edited_alt_text.assert_called_once_with(
        session_id = "1",
        entry_id="101",
        edited_alt_text="new changed text"
    )

    # verify correct error is returned when Supabase cant save
    assert result == "UNABLE_TO_SAVE"

# EDIT-UT3: Tests that a request missing entry_id returns the appropriate error
def test_edit_alt_text_invalid_missing_entry_id():
    # mock AltTextHistory client as its not being tested
    mock_history = MagicMock()
    request = MagicMock()
    request.body = json.dumps(
        {
            "edited_alt_text":"new changed text"
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request, history=mock_history)

    # verify correct error is returned when alt text is missing
    assert result == "INVALID_REQUEST"
    mock_history.return_value.update_edited_alt_text.assert_not_called()


# EDIT-UT4: Tests that that a request missing alt text returns the appropriate error
def test_edit_alt_text_invalid_missing_alt_text():
    # mock AltTextHistory client as its not being tested
    mock_history = MagicMock()
    request = MagicMock()
    request.body = json.dumps(
        {
            "entry_id": "101",
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request, history=mock_history)

    # verify correct error is returned when alt text is missing
    assert result == "INVALID_REQUEST"
    mock_history.return_value.update_edited_alt_text.assert_not_called()

# EDIT-UT5: Tests that when Supabase throws an exception that the appropriate error is returned
def test_edit_alt_text_exception_thrown():
    # mock AltTextHistory client as its not being tested
    mock_history = MagicMock()
    mock_history.update_edited_alt_text.side_effect = Exception("SUPABASE ERROR")

    request = MagicMock()
    request.body = json.dumps(
        {
            "edited_alt_text":"new changed text",
            "entry_id": "101",
        }
    )
    request.session_id = "1"

    result = edit_alt_text(request, history=mock_history)
    # verify correct error is returned when exception is thrown
    assert result == "UNABLE_TO_SAVE"
