import base64
import pandas as pd
import uuid
from databases.connect_supabase import supabase

class GenAltText:
    def __init__(self):
        pass
    
    def trigger_model(self,image,session_id):
        # needs to be changed to trigger real model 
        mock_alt_text = uuid.uuid4().hex

        # after alt text has been successfully generated, the alt text and image is saved to the history 
        self.insert_history(image,mock_alt_text, session_id)

        # returns alt text to show user
        return mock_alt_text
    

    def insert_history(self, image, alt_text, session_id):

        # need to convert bytes to a string in order to send to superbase with json
        image_bytes = image.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        #dont need to give time stamp it will by default use current time
        supabase.table("history").insert({
            "session_id":session_id,
            "image":image_b64,
            "alt_text": alt_text
        }).execute()
        

    # this function is for testing purposes to store images from database
    def save_image(self,entry_id=1):
        response = (
            supabase.table("history")
            .select("image")
            .eq("entry_id",entry_id)
            .single()
            .execute()
        )
        # the image is stored in the databases as string need to convert to bytes
        image_b64 = response.data["image"]
        image_bytes = base64.b64decode(image_b64)

        with open("testing_saving_img_from_db.png", "wb") as f:
            f.write(image_bytes)
