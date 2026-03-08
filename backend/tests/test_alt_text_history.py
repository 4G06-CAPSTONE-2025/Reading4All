from unittest.mock import MagicMock

from services.alt_text_history import AltTextHistory



def test_get_alt_text_history_one_entry():

    mock_supabase = MagicMock()
    mock_response = MagicMock()

    mock_response.data = [
        {
            "image": "test_image.png",
            "alt_text": "generated alt text",
            "edited_alt_text": None
        }
    ]
    query = mock_supabase.table().select().eq().order().limit()
    query.execute.return_value = mock_response
    get_history_service = AltTextHistory(supabase=mock_supabase)


    history = get_history_service.get_alt_text_history("test_session_id")

    assert len(history) == 1
    assert history[0]["image"] == "test_image.png"
    assert history[0]["altText"] == "generated alt text"
    get_history_service = AltTextHistory(supabase=mock_supabase)


    history = get_history_service.get_alt_text_history("test_session_id")

    assert len(history) == 1
    assert history[0]["image"] == "test_image.png"
    assert history[0]["altText"] == "generated alt text"


def test_get_alt_text_history_zero_entries():

    mock_supabase = MagicMock()
    mock_response = MagicMock()

    mock_response.data = []

    query = mock_supabase.table().select().eq().order().limit()
    query.execute.return_value = mock_response
    get_history_service = AltTextHistory(supabase=mock_supabase)

    history = get_history_service.get_alt_text_history("test_session_id")

    assert len(history) == 0


def test_get_alt_text_history_edited_entry():

    mock_supabase = MagicMock()
    mock_response = MagicMock()

    mock_response.data = [
        {
            "image": "test_image.png",
            "alt_text": "generated alt text",
            "edited_alt_text": "edited alt text"
        }
    ]
    query = mock_supabase.table().select().eq().order().limit()
    query.execute.return_value = mock_response
    get_history_service = AltTextHistory(supabase=mock_supabase)

    history = get_history_service.get_alt_text_history("test_session_id")

    assert len(history) == 1
    assert history[0]["image"] == "test_image.png"
    assert history[0]["altText"] == "edited alt text"


def test_get_alt_text_history_null_string_edited_entry():

    mock_supabase = MagicMock()
    mock_response = MagicMock()

    mock_response.data = [
        {
            "image": "test_image.png",
            "alt_text": "generated alt text",
            "edited_alt_text": "NULL"
        }
    ]
    query = mock_supabase.table().select().eq().order().limit()
    query.execute.return_value = mock_response
    get_history_service = AltTextHistory(supabase=mock_supabase)

    history = get_history_service.get_alt_text_history("test_session_id")

    assert len(history) == 1
    assert history[0]["image"] == "test_image.png"
    assert history[0]["altText"] == "generated alt text"


def test_get_alt_text_history_multiple_entries():

    mock_supabase = MagicMock()
    mock_response = MagicMock()

    mock_response.data = [
        {
            "image": "test_image1.png",
            "alt_text": "generated alt text 1",
            "edited_alt_text": None
        },
        {
            "image": "test_image2.png",
            "alt_text": "generated alt text 2",
            "edited_alt_text": "edited alt text 2"
        }
    ]
    query = mock_supabase.table().select().eq().order().limit()
    query.execute.return_value = mock_response
    get_history_service = AltTextHistory(supabase=mock_supabase)

    history = get_history_service.get_alt_text_history("test_session_id")

    assert len(history) == 2
    # first entry should have no edits
    assert history[0]["image"] == "test_image1.png"
    assert history[0]["altText"] == "generated alt text 1"

    # second entry should show edited alt text
    assert history[1]["image"] == "test_image2.png"
    assert history[1]["altText"] == "edited alt text 2"
