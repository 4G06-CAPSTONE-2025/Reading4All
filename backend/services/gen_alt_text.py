
import mysql.connector
import pandas as pd
import uuid

def connect_to_sql():
    connection = mysql.connector.connect(
        host = 'localhost', # add values here if needed
        user = 'root', 
        password = ''
    )
    return connection

class GenAltText:
    def __init__(self):
        pass
    
    def trigger_model(self,image):
        # needs to be changed to trigger real model 
        mock_alt_text = uuid.uuid4().hex

        # after alt text has been successfully generated, the alt text and image is saved to the history 
        GenAltText.insert_history(image,mock_alt_text)

        # returns alt text to show user
        return mock_alt_text
    

    def insert_history(image, alt_text):
        # setup 
        connection = connect_to_sql()
        cursor= connection.cursor()
        cursor.execute("USE reading4allDB")

        #inserting into the db, users uploaded image and associated alt text
        cursor.execute("INSERT INTO history (image, alt_text) VALUES (%s,%s)", (image.read(),alt_text))

        connection.commit()
        cursor.close()
        connection.close()


    # this function is for testing purposes to store images from database
    def save_image(id):
        connection = connect_to_sql()

        cursor= connection.cursor()
        cursor.execute("USE reading4allDB")
        cursor = connection.cursor()
        cursor.execute("SELECT image FROM history WHERE id = %s", (id,)) # default sql behaviour is to increment id from 1 not0 
        image_bytes = cursor.fetchall()[0][0]

        cursor.close()
        connection.close()

        with open("default_image_name2027.png", "wb") as f:
            f.write(image_bytes)

