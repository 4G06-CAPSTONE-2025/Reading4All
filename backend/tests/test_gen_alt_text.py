from unittest.mock import MagicMock, patch
from io import BytesIO
from services.gen_alt_text import GenAltText

@patch("services.gen_alt_text.get_supabase_admin_client")
def test_post_process_alt_text_extra_spaces(mock_supabase):
    mock_supabase.return_value = MagicMock()

    cleanup = GenAltText()
    input_text = "     Text  here.  "

    result = cleanup.post_process_alt_text(input_text)
    assert result == "Text here."


@patch("services.gen_alt_text.get_supabase_admin_client")
def test_post_process_alt_text_repeat_words(mock_supabase):
    mock_supabase.return_value = MagicMock()

    cleanup = GenAltText()
    input_text = "Repeat Repeat here."

    result = cleanup.post_process_alt_text(input_text)
    assert result == "Repeat here."


@patch("services.gen_alt_text.get_supabase_admin_client")
def test_post_process_alt_text_lower_case(mock_supabase):
    mock_supabase.return_value = MagicMock()

    cleanup = GenAltText()
    input_text = "lowercase  here."

    result = cleanup.post_process_alt_text(input_text)
    assert result == "Lowercase here."


@patch("services.gen_alt_text.get_supabase_admin_client")
def test_post_process_alt_text_missing_period(mock_supabase):
    mock_supabase.return_value = MagicMock()

    cleanup = GenAltText()
    input_text = "Missing period"

    result = cleanup.post_process_alt_text(input_text)
    assert result == "Missing period."


@patch("services.gen_alt_text.get_supabase_admin_client")
def test_post_process_alt_text_multiple_sentences(mock_supabase):
    mock_supabase.return_value = MagicMock()

    cleanup = GenAltText()
    input_text = "this is the first sentence. second here."

    result = cleanup.post_process_alt_text(input_text)
    assert result == "This is the first sentence. Second here."


@patch("services.gen_alt_text.requests.post")
@patch("services.gen_alt_text.get_supabase_admin_client")
def test_trigger_model_no_alt_text(mock_supabase, mock_post):
    mock_supabase.return_value = MagicMock()

    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_post.return_value = mock_response

    image = BytesIO(b"Fake Image")
    image.content_type = "image/png"

    gen = GenAltText()
    result = gen.trigger_model(image,"1")

    assert result == (None, None)


@patch("services.gen_alt_text.requests.post")
@patch("services.gen_alt_text.get_supabase_admin_client")
def test_trigger_model_success(mock_supabase, mock_post):
    mock_supabase.return_value = MagicMock()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "alt_text":"Physics Diagram: Circuit description here."
    }
    mock_post.return_value = mock_response
    image = BytesIO(b"Fake Image")
    image.content_type = "image/png"

    gen = GenAltText()
    gen.insert_history = MagicMock(return_value="101")

    result = gen.trigger_model(image,"1")

    gen.insert_history.assert_called_once_with(
        b"Fake Image",
        "Circuit description here.",
        "1"
    )
    assert result == ("Circuit description here.", "101")
