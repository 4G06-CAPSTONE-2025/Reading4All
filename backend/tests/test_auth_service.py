"""
Author: Moly Mikhail
Date: March 2026
Purpose: Unit tests for the AuthenticationService module,
specifically verifying user signup behavior.
"""

from unittest.mock import MagicMock
from services.auth_service import AuthService



# AUTH-UT1: Tests successful user signup with valid email and password
def test_signup_user_success():
    # mock supabase client as its needed to initialize AuthService
    mock_supabase = MagicMock()

    # mocking a user, session to be used for supabase response
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


# AUTH-UT2: Tests user signup with invalid email format (not @mcmaster.ca)
def test_signup_invalid_email():

    # mock supabase client as its needed to initialize AuthService
    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)

    # checks that an error is raised for invalid email formats
    try:
        auth_service.signup("student@gmail.com", "passwordABC123")
        assert False, "Value Error should be raised when email is not @mcmaster.ca"
    except ValueError as e:
        assert str(e) == "Email must be a @mcmaster.ca address"
    mock_supabase.auth.sign_up.assert_not_called()


# AUTH-UT3: Tests user signup with password shorter than 8 characters
def test_signup_short_password():
    # mock supabase client as its needed to initialize AuthService
    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)

    # checks that an error is raised for invalid passwords
    try:
        auth_service.signup("student@mcmaster.ca", "short")
        assert False, "Value Error should be raised when password is too short"
    except ValueError as e:
        assert str(e) == "Password must be at least 8 characters"

    mock_supabase.auth.sign_up.assert_not_called()

# AUTH-UT4: Tests user signup with empty email and password
def test_signup_empty_email_and_password():
    # mock supabase client as its needed to initialize AuthService
    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)

    # checks that an error is raised for empty email & passwords
    try:
        auth_service.signup("", "")
        assert False, "Value Error should be raised when email and password are empty"
    except ValueError as e:
        assert str(e) == "Email must be a @mcmaster.ca address"

    mock_supabase.auth.sign_up.assert_not_called()

# AUTH-UT4.1: Tests user signup with empty email and password
def test_signup_none_email_and_password():
    # mock supabase client as its needed to initialize AuthService
    mock_supabase = MagicMock()
    auth_service = AuthService(supabase=mock_supabase)
    
    # checks that an error is raised for empty email
    try:
        auth_service.signup(None, None)
        assert False, "Value Error should be raised when email and password are None"
    except ValueError as e:
        assert str(e) == "Email must be a @mcmaster.ca address"

    mock_supabase.auth.sign_up.assert_not_called()

# AUTH-UT5: Tests user signup with capital letters in email
def test_signup_capital_letter_email():

    # mock supabase client as its needed to initialize AuthService
    mock_supabase = MagicMock()

    mock_supabase.auth.sign_up.return_value = MagicMock()

    auth_service = AuthService(supabase=mock_supabase)
    auth_service.signup("STUDENT@MCMASTER.CA", "passwordABC123")

    # checks that signup is called with converted lowercase email 
    mock_supabase.auth.sign_up.assert_called_once_with(
        {
            "email": "student@mcmaster.ca",
            "password": "passwordABC123"
        }
    )

# AUTH-UT6: Tests user signup with email that is already in use.
def test_signup_with_existing_email():

    # mock supabase client as its needed to initialize AuthService
    mock_supabase = MagicMock()

    # mocking a user, session to be used for supabase response
    mock_user = MagicMock()
    mock_user.id = "this user id already exists"

    mock_response = MagicMock()
    mock_response.user = mock_user
    mock_response.session = None

    mock_supabase.auth.sign_up.return_value = mock_response

    auth_service = AuthService(supabase=mock_supabase)
    # checks that users signing up with an existing email get an error
    try:
        auth_service.signup("student@mcmaster.ca", "passwordABC123")
        assert False, "Value Error should be raised when email is already in use"
    except ValueError as e:
        assert str(e) == "Email is already in use"
