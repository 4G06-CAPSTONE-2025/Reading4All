This guide explains how to set up your local environment and run the **same checks** that our GitHub Actions **Backend CI** workflow runs.

## What “CI-equivalent” means

Running the local CI-equivalent build will:

1. Run **Pylint** (lint/style checks)
2. Run **Django system checks** (`python manage.py check`)
3. Run **Django tests** (`python manage.py test`)

If all three pass locally, your backend changes should pass CI.

---

## Prerequisites

- **Python 3.12+** recommended (CI uses Python 3.12)
- macOS / Linux / Windows supported
- You should be in the project repo that contains the `backend/` directory

---

## 1) First-time setup

### macOS / Linux

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## 2) Run using:
./ci-local.sh

# What ci-local.sh does

# It sets the same environment variables CI uses, then runs:

python -m pylint **/*.py

python manage.py check

python manage.py test