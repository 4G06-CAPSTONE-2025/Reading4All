from databases.connect_supabase import get_supabase_admin_client

"""
Author: Moly Mikhail
Date: Jan 2026
Purpose: Retrieves and updates alt text history entries stored in Supabase for a user session. 
"""
class AltTextHistory:

    def __init__(self, supabase=None):
        self.max_entries = 10
        self.supabase = supabase or get_supabase_admin_client()

    def get_alt_text_history(self, session_id):

        # gets the most recent alt text entries for a given session
        results_history = (
            self.supabase.table("history")
            .select("image, alt_text, edited_alt_text")
            .eq("session_id", session_id)
            .order("time_gen", desc=True)
            .limit(self.max_entries)
            .execute()
        )

        history = []

        for entry in results_history.data:
            # uses edited alt text if available, if not uses the original alt text
            if entry["edited_alt_text"] not in (None, "NULL"):
                alt_text_to_show = entry["edited_alt_text"]
            else:
                alt_text_to_show = entry["alt_text"]

            entry = {
                "image": entry["image"],
                "altText": alt_text_to_show
                }
            history.append(entry)
        return history

    def update_edited_alt_text(self, session_id, entry_id, edited_alt_text):
        try:
            # Gets the specific entry for a given session
            # Ensures entry is found before making the update
            response = (
                self.supabase.table("history")
                .select("entry_id")
                .eq("session_id", session_id)
                .eq("entry_id", entry_id)
                .order("time_gen", desc=True)
                .limit(1)
                .execute()
            )

            # specific entry is not found so update cant be made
            if not response.data:
                return False

            entry_id = response.data[0]["entry_id"]

            # updates the edited_alt_text col in database 
            self.supabase.table("history").update(
                {"edited_alt_text": edited_alt_text}
            ).eq("entry_id", entry_id).execute()

            return True

        except Exception as e: # pylint: disable=broad-exception-caught
            return False
