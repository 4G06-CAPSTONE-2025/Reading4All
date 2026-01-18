
import mysql.connector
import pandas as pd

def connect_to_sql():
    connection = mysql.connector.connect(
        host = 'localhost', # add values here if needed
        user = 'root', 
        password = ''
    )
    return connection

class AltTextHistory: 

    def __init__(self):
        max_entries = 10
    
    def get_alt_text_history(self):

        connection = connect_to_sql()

        cursor= connection.cursor()
        cursor.execute("USE reading4allDB")


        cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 10")
        results_history = cursor.fetchall()

        # for testing purposes: 
        # for index in range(len(results_history)):
        #     print("Entry:", index)
        #     #This prints the mock alt text that is saved
        #     print(results_history[index][2])

        history = {"imageBytes":[], "altText":[]}

        for entry in range(len(results_history)):
            history["imageBytes"].append(results_history[entry][1])
            history["altText"].append(results_history[entry][2])
        

        connection.commit()
        cursor.close()
        connection.close()
        
        return history



        