set -e

export DJANGO_SETTINGS_MODULE=backend.settings
export SECRET_KEY=local-secret
export DEBUG=0

echo "▶ Running pylint"
pylint **/*.py

echo "▶ Running Django checks"
python manage.py check

echo "▶ Running tests"
python manage.py test

echo "✅ CI-equivalent checks passed"