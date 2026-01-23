from databases.connect_supabase import supabase


class AltTextHistory:

    def __init__(self):
        self.max_entries = 10

    def get_alt_text_history(self, session_id):
        results_history = (
            supabase.table("history")
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
