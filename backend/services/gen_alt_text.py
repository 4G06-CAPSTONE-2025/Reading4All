"""
Author: Moly Mikhail and Casey Francine Bulaclac
Date: Jan 2026
Purpose: Calls the AI model on Hugging Face, post-processes the 
generated alt text to clean up the format and structure. 
Finally, stores the generated output in Supabase.
"""

import base64
import os
import re
import requests

from databases.connect_supabase import get_supabase_admin_client


class GenAltText:
    def __init__(self, supabase=None):
        self.supabase = supabase or get_supabase_admin_client()
        self.max_entries = 10
        self.hf_token = os.getenv("HUGGINGFACE_READ_TOKEN")
        # pylint: disable=line-too-long
        self.hf_url = "https://adexn5i2xzlhi5pn.us-east-1.aws.endpoints.huggingface.cloud"


    def trigger_model(self, image, session_id):

        # sends image as bytes to HuggingFace model to generate alt text
        image_bytes = image.read()

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": image.content_type,
        }
        response = requests.post(
            self.hf_url,
            headers=headers,
            data=image_bytes,
            timeout=180
        )
        raw_alt_text = response.json().get('alt_text')


        # if no alt text is returned by model
        if not raw_alt_text:
            return None, None

        # removing prompt used in model from alt-text returned
        # which is stated before the the colon
        raw_alt_text = raw_alt_text.split(":", 1)[1].strip()

        # clean up post processing layer
        post_alt_text = self.post_process_alt_text(raw_alt_text)

        # after alt text has been successfully generated, the cleaned up alt text
        # and image is saved to the history
        entry_id = self.insert_history(image_bytes, post_alt_text, session_id)

        return post_alt_text, entry_id



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
            return None

        return response.data[0]["entry_id"]


    def remove_oldest_entry(self, oldest_entry_id):

        # deletes the row with the found entry_id
        (
            self.supabase.table("history")
            .delete()
            .eq("entry_id", oldest_entry_id)
            .execute()
        )


    def insert_history(self, image_bytes, alt_text, session_id):

        # checks if session entries stored is at limit
        if self.session_entries_count(session_id) == self.max_entries:

            # finds oldest entry in history
            oldest_entry_id = self.find_oldest_entry(session_id)

            # removes oldest entry to make room for new entry
            if oldest_entry_id:
                self.remove_oldest_entry(oldest_entry_id)

        # need to convert bytes to a string in order to store in Supabase
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # dont need to give time stamp it will by default use current time
        response = (
                self.supabase.table("history").insert(
                    {
                        "session_id": session_id,
                        "image": image_b64, "alt_text":
                        alt_text, "edited_alt_text": "NULL"
                    }
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

    # cleans up raw alt text
    def post_process_alt_text(self, alt_text):

        # removes leading/trailing whitespace
        alt_text = alt_text.strip()

        # replaces multiple spaces (or tabs/newlines) with a single space
        alt_text = re.sub(r"\s+", " ", alt_text)

        # remove repeated consecutive words, case-insensitive
        alt_text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", alt_text, flags=re.IGNORECASE)

        # remove repeated punctuations
        alt_text = re.sub(r"([.!?,;:]){2,}", r"\1", alt_text)

        # capitalize first character if needed
        if alt_text and alt_text[0].islower():
            alt_text = alt_text[0].upper() + alt_text[1:]

        # ensure sentence ends with punctuation
        if alt_text and alt_text[-1] not in ".!?":
            alt_text += "."

        # capitalize the first letter of each sentence
        sentences = re.split(r'([.!?]\s*)', alt_text)  # keep punctuation
        capitalized_sentences = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
            capitalized_sentences.append(sentence)

            # add back punctuation if it exists
            if i + 1 < len(sentences):
                capitalized_sentences.append(sentences[i + 1])

        alt_text = "".join(capitalized_sentences)

        return alt_text
