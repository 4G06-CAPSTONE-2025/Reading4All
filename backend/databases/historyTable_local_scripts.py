import mysql.connector

def connect_to_sql():
    connection = mysql.connector.connect(
        host = 'localhost', # add values here if needed
        user = 'root', 
        password = ''
    )
    return connection


def delete_entries():
    connection = connect_to_sql()

    cursor= connection.cursor()
    cursor.execute("USE reading4allDB")
    cursor.execute("DELETE FROM history;")
    connection.commit()
    cursor.close()
    connection.close()



def create_db():
    connection = connect_to_sql()

    cursor= connection.cursor()

    cursor.execute("CREATE DATABASE IF NOT EXISTS reading4allDB")
    cursor.execute("USE reading4allDB")

    cursor.execute("CREATE TABLE IF NOT EXISTS history (entry_id INT AUTO_INCREMENT PRIMARY KEY, session_id INT NOT NULL, image LONGBLOB NOT NULL, alt_text LONGTEXT NOT NULL)")
    connection.commit()
    cursor.close()
    connection.close()

create_db()