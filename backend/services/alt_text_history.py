from databases.connect_supabase import get_supabase_admin_client


class AltTextHistory:

    def __init__(self):
        self.max_entries = 10
        self.supabase = get_supabase_admin_client()

    def get_alt_text_history(self, session_id):
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

            if entry["edited_alt_text"] is not None:
                altTextToShow = entry["edited_alt_text"]
            else:
                altTextToShow = entry["alt_text"]

            entry = {
                "image": entry["image"],
                "altText": altTextToShow
                }
            history.append(entry)
        return history

    def update_edited_alt_text(self, session_id, entry_id, edited_alt_text):
        try:
            # get most recent history entry for this session
            # edits only apply to latest entry (current generated alt text
            response = (
                self.supabase.table("history")
                .select("entry_id")
                .eq("session_id", session_id)
                .eq("entry_id", entry_id)
                .order("time_gen", desc=True)
                .limit(1)
                .execute()
            )

            if not response.data:
                return False

            entry_id = response.data[0]["entry_id"]

            # updates the edited_alt_text col
            self.supabase.table("history").update(
                {"edited_alt_text": edited_alt_text}
            ).eq("entry_id", entry_id).execute()

            return True

        except Exception as e: # pylint: disable=broad-exception-caught
            print("UPDATE EDITED ALT TEXT ERROR:", e)
            return False
