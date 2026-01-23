import mysql.connector

<<<<<<< HEAD
def connect_to_sql():
    connection = mysql.connector.connect(
        host = 'localhost', # add values here if needed
        user = 'root', 
        password = ''
=======

def connect_to_sql():
    connection = mysql.connector.connect(
        host="localhost", user="root", password=""  # add values here if needed
>>>>>>> dev
    )
    return connection


def delete_entries():
    connection = connect_to_sql()

<<<<<<< HEAD
    cursor= connection.cursor()
=======
    cursor = connection.cursor()
>>>>>>> dev
    cursor.execute("USE reading4allDB")
    cursor.execute("DELETE FROM history;")
    connection.commit()
    cursor.close()
    connection.close()


<<<<<<< HEAD

def create_db():
    connection = connect_to_sql()

    cursor= connection.cursor()
=======
def create_db():
    connection = connect_to_sql()

    cursor = connection.cursor()
>>>>>>> dev

    cursor.execute("CREATE DATABASE IF NOT EXISTS reading4allDB")
    cursor.execute("USE reading4allDB")

<<<<<<< HEAD
    cursor.execute("CREATE TABLE IF NOT EXISTS history (entry_id INT AUTO_INCREMENT PRIMARY KEY, session_id INT NOT NULL, image LONGBLOB NOT NULL, alt_text LONGTEXT NOT NULL)")
=======
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS history"
        " (entry_id INT AUTO_INCREMENT PRIMARY KEY, session_id INT NOT NULL, " \
        "image LONGBLOB NOT NULL," \
        " alt_text LONGTEXT NOT NULL)"
    )
>>>>>>> dev
    connection.commit()
    cursor.close()
    connection.close()

<<<<<<< HEAD
create_db()
=======

create_db()
>>>>>>> dev
