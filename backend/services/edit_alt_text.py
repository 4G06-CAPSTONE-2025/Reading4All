import json
from services.alt_text_history import AltTextHistory

# get history service to interact with Supabase history table 
history = AltTextHistory()


def edit_alt_text(request):
    try:
        # parse request 
        body = json.loads(request.body)

        image = body.get("image")
        edited_alt_text = body.get("edited_alt_text")

        # same as in GET_alt_test_hisotry.py
        session_id = 2026

        if not image or not edited_alt_text:
            return "INVALID_REQUEST"

        # edits the edited_alt_text col in the last row of history table
        success = history.update_edited_alt_text(
            session_id=session_id,
            image=image,
            edited_alt_text=edited_alt_text
        )

        if not success:
            return "UNABLE_TO_SAVE"

        return "Success"

    except Exception as e:
        print("EDIT ALT TEXT CONTROLLER ERROR:", e)
        return "UNABLE_TO_SAVE"
