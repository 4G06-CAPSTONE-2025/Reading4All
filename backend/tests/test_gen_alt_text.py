"""
Author: Moly Mikhail
Date: March 2026
Purpose: Unit tests for the generating alt text, 
specifically the clean up post process 
and handling output from the AI Model.
"""

from unittest.mock import MagicMock, patch
from io import BytesIO
from services.gen_alt_text import GenAltText

def test_post_process_alt_text_extra_spaces():
    # mock supabase client as its needed to initialize GenAltTex
    mock_supabase = MagicMock()

    cleanup = GenAltText(supabase=mock_supabase)
    input_text = "     Text  here.  "

    # checks that extra spaces are removed
    result = cleanup.post_process_alt_text(input_text)
    assert result == "Text here."


def test_post_process_alt_text_repeat_words():
    # mock supabase client as its needed to initialize GenAltTex
    mock_supabase = MagicMock()


    cleanup = GenAltText(supabase=mock_supabase)
    input_text = "Repeat Repeat here."

    # checks that repeat words are removed
    result = cleanup.post_process_alt_text(input_text)
    assert result == "Repeat here."


def test_post_process_alt_text_lower_case():
    # mock supabase client as its needed to initialize GenAltTex
    mock_supabase = MagicMock()

    cleanup = GenAltText(supabase=mock_supabase)
    input_text = "lowercase  here."

    # checks that sentences being with uppercase letters
    result = cleanup.post_process_alt_text(input_text)
    assert result == "Lowercase here."


def test_post_process_alt_text_missing_period():
    # mock supabase client as its needed to initialize GenAltTex
    mock_supabase = MagicMock()

    cleanup = GenAltText(supabase=mock_supabase)
    input_text = "Missing period"

    # checks that missing punctuation is added
    result = cleanup.post_process_alt_text(input_text)
    assert result == "Missing period."


def test_post_process_alt_text_multiple_sentences():
    # mock supabase client as its needed to initialize GenAltTex
    mock_supabase = MagicMock()

    cleanup = GenAltText(supabase=mock_supabase)
    input_text = "this is the first sentence. second here."

    # checks multiple sentences for capitalization
    result = cleanup.post_process_alt_text(input_text)
    assert result == "This is the first sentence. Second here."

# mocks API post to hugging face AI Model
@patch("services.gen_alt_text.requests.post")
def test_trigger_model_no_alt_text(mock_post):
    # mock supabase client as its needed to initialize GenAltTex
    mock_supabase = MagicMock()

    # mock HuggingFace API Response
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_post.return_value = mock_response

    image = BytesIO(b"Fake Image")
    image.content_type = "image/png"

    # checks if model returns no alt text that the result is also none
    gen = GenAltText(supabase=mock_supabase)
    result = gen.trigger_model(image,"1")

    assert result == (None, None)

# mocks API post to hugging face AI Model
@patch("services.gen_alt_text.requests.post")
def test_trigger_model_success(mock_post):
    # mock supabase client as its needed to initialize GenAltTex
    mock_supabase = MagicMock()

    # mock HuggingFace API Response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "alt_text":"Physics Diagram: Circuit description here."
    }
    mock_post.return_value = mock_response
    image = BytesIO(b"Fake Image")
    image.content_type = "image/png"

    gen = GenAltText(supabase=mock_supabase)
    gen.insert_history = MagicMock(return_value="101")

    result = gen.trigger_model(image,"1")

    gen.insert_history.assert_called_once_with(
        b"Fake Image",
        "Circuit description here.",
        "1"
    )

    # checks that AI model generated text is returned as the result
    assert result == ("Circuit description here.", "101")
