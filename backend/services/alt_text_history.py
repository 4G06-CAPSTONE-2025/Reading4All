from databases.connect_supabase import get_supabase_admin_client


class AltTextHistory:

    def __init__(self):
        self.max_entries = 10
        self.supabase = get_supabase_admin_client()

    def get_alt_text_history(self, session_id):
        results_history = (
            self.supabase.table("history")
            .select("image, alt_text")
            .eq("session_id", session_id)
            .order("time_gen", desc=True)
            .limit(self.max_entries)
            .execute()
        )

        history = []

        for entry in results_history.data:

            entry = {
                "image": entry["image"],
                "altText": entry["alt_text"]
                }
            history.append(entry)
        return history
