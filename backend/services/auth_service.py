from django.contrib.auth import get_user_model

User = get_user_model()

class AuthService:
    def signup(self, email: str, password: str):
        email = (email or "").strip().lower()

        if not email.endswith("@mcmaster.ca"):
            raise ValueError("Email must be a @mcmaster.ca address")

        if len(password or "") < 8:
            raise ValueError("Password must be at least 8 characters")

        # default Django user: username is required; use email as username
        if User.objects.filter(username=email).exists():
            raise ValueError("User already exists")

        user = User.objects.create_user(username=email, email=email, password=password)
        return user
