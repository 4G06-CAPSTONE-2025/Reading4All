from unittest.mock import MagicMock
from services.auth_service import AuthService




def test_signup_user_success():

    mock_supabase = MagicMock()

    mock_user = MagicMock()
    mock_user.id = "test_user_id"

    mock_session = MagicMock()
    mock_session.access_token = "test_access_token"

    mock_response = MagicMock()
    mock_response.user = mock_user
    mock_response.session = mock_session

    mock_supabase.auth.sign_up.return_value = mock_response

    auth_service = AuthService(supabase=mock_supabase)
    result = auth_service.signup("student@mcmaster.ca", "passwordABC123")
    assert result == {
        "user_id": "test_user_id",
        "access_token": "test_access_token"
    }

def test_signup_invalid_email():

    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)

    try:
        auth_service.signup("student@gmail.com", "passwordABC123")
        assert False, "Value Error should be raised when email is not @mcmaster.ca"
    except ValueError as e:
        assert str(e) == "Email must be a @mcmaster.ca address"
    mock_supabase.auth.sign_up.assert_not_called()


def test_signup_short_password():

    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)

    try:
        auth_service.signup("student@mcmaster.ca", "short")
        assert False, "Value Error should be raised when password is too short"
    except ValueError as e:
        assert str(e) == "Password must be at least 8 characters"
    mock_supabase.auth.sign_up.assert_not_called()


def test_signup_empty_email_and_password():

    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)

    try:
        auth_service.signup("", "")
        assert False, "Value Error should be raised when email and password are empty"
    except ValueError as e:
        assert str(e) == "Email must be a @mcmaster.ca address"
    mock_supabase.auth.sign_up.assert_not_called()


def test_signup_none_email_and_password():

    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)

    try:
        auth_service.signup(None, None)
        assert False, "Value Error should be raised when email and password are None"
    except ValueError as e:
        assert str(e) == "Email must be a @mcmaster.ca address"
    mock_supabase.auth.sign_up.assert_not_called()


def test_signup_capital_letter_email():

    mock_supabase = MagicMock()

    mock_supabase.auth.sign_up.return_value = MagicMock()

    auth_service = AuthService(supabase=mock_supabase)
    auth_service.signup("STUDENT@MCMASTER.CA", "passwordABC123")
    mock_supabase.auth.sign_up.assert_called_once_with(
        {
            "email": "student@mcmaster.ca",
            "password": "passwordABC123"
        }
    )
