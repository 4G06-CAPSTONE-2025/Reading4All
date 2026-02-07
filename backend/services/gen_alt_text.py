import base64
import os
import requests

from databases.connect_supabase import get_supabase_admin_client


class GenAltText:
    def __init__(self):
        self.supabase = get_supabase_admin_client()
        self.max_entries = 10
        self.hf_token = os.getenv("HUGGINGFACE_READ_TOKEN")
        self.hf_url = """
            https://hdzn5l02irp5ygnw.us-east-1.aws.endpoints.huggingface.cloud/"
        """


    def trigger_model(self, image, session_id):

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "image/png"   # or image/jpeg
        }
        response = requests.post(
            self.hf_url,
            headers=headers,
            data=image,
            timeout=30
        )

        print(response.json())
        alt_text = response.json()[0]['alt_text']



        # returns alt text to show user
        if not alt_text:
            return None, None
        # after alt text has been successfully generated, the alt text
        # and image is saved to the history
        entry_id = self.insert_history(image, alt_text, session_id)
        return alt_text, entry_id



    def session_entries_count(self, session_id):

        # before adding new entires to the history table,
        # must check if entries exceed max amt per session

        # returns the amount of entries for the inputted session id
        response = (
            self.supabase.table("history")
            .select('*', count = 'exact')
            .eq("session_id", session_id)
            .execute()
        )

        return response.count


    def find_oldest_entry(self, session_id):

        # this retrieves the oldest entry_id for the inputted session id
        response = (
            self.supabase.table("history")
            .select("entry_id")
            .eq("session_id", session_id)
            .order("time_gen", desc=False)
            .limit(1)
            .execute()
        )

        if not response.data:
            return False

        return response.data[0]["entry_id"]


    def remove_oldest_entry(self, oldest_entry_id):

        # deletes the row with the found entry_id
        (
            self.supabase.table("history")
            .delete()
            .eq("entry_id", oldest_entry_id)
            .execute()
        )


    def insert_history(self, image, alt_text, session_id):

        if self.session_entries_count(session_id) == self.max_entries:

            oldest_entry_id = self.find_oldest_entry(session_id)

            if oldest_entry_id:
                self.remove_oldest_entry(oldest_entry_id)

        # need to convert bytes to a string in order to send to superbase with json
        image_bytes = image.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # dont need to give time stamp it will by default use current time
        response = (
                self.supabase.table("history").insert(
                    {"session_id": session_id,
                    "image": image_b64, "alt_text":
                    alt_text, "edited_alt_text": "NULL"}
                    ).execute()
                )
        return response.data[0]["entry_id"]

    # this function is for testing purposes to store images from database
    def save_image(self, entry_id=1):
        response = (
            self.supabase.table("history")
            .select("image")
            .eq("entry_id", entry_id)
            .single()
            .execute()
        )
        # the image is stored in the databases as string need to convert to bytes
        image_b64 = response.data["image"]
        image_bytes = base64.b64decode(image_b64)

        with open("testing_saving_img_from_db.png", "wb") as f:
            f.write(image_bytes)
