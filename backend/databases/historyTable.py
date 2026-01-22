import mysql.connector

def connect_to_sql():
    connection = mysql.connector.connect(
        host = 'localhost', # add values here if needed
        user = 'root',
        password = ''
    )
    return connection

def create_db():
    connection = connect_to_sql()

    cursor= connection.cursor()

    cursor.execute("CREATE DATABASE IF NOT EXISTS reading4allDB")
    cursor.execute("USE reading4allDB")

    cursor.execute("CREATE TABLE IF NOT EXISTS history "
    "(id INT AUTO_INCREMENT PRIMARY KEY, image LONGBLOB NOT NULL," \
    " alt_text LONGTEXT NOT NULL)")
    connection.commit()
    cursor.close()
    connection.close()

create_db()
