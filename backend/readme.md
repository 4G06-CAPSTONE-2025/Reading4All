# Running Backend Locally and Testing on Postman

### Setup .env or env variables in terminal

To run in terminal: 

```bash
export SUPABASE_URL=https://nrukxjmzrkjepsdpbmhd.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
```

To use a `.env` file
- Create a .`env` file in the `backend/` directory 
- Follow this format: 
```bash
SUPABASE_URL = https://nrukxjmzrkjepsdpbmhd.supabase.co
SUPABASE_SERVICE_ANON_KEY = <your-service-role-key>
```


### Running the backend 

1. Ensure all requirements are installed. Run these commands in terminal: 
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py migrate
```
2. Start Server:
```bash
python manage.py runserver
```

### Testing on Postman 
1. Ensure you have Postman installed. 

#### For new users: 
1. Must run `POST /api/send-verification/`
    -   In Postman:
        -   set the API Method to `POST`
        -   set the URL to `http://127.0.0.1:8000/api/send-verification/`
        -   set the body type to `raw`
            -   Input this JSON: 
            ```py
            {
            "email": "<mcmaster_email_goes_here>",
            "password": "<password_goes_here>"
            }
            ```
        -   Send request
            -   If request is successful you will get a status 200!
    -   Check your inputted McMaster email to obtain a OTP code, note this code down! 
2. Must run `POST /api/verify/` 
    -   In Postman:
        -   set the API Method to `POST`
        -   set the URL to `http://127.0.0.1:8000/api/verify/`
        -   set the body type to `raw`
            -   Input this JSON: 
            ```py
            {
            "email": "<mcmaster_email_goes_here>",
            "token": "<token_from_email_goes_here>",
            "password": "<password_goes_here>"
            }
            ```
        -   Send request
            -   If request is successful you will get a status 200!
3. Must run `POST /api/login/` 
   -   In Postman:
        -   set the API Method to `POST`
        -   set the URL to `http://127.0.0.1:8000/api/login/`
        -   set the body type to `raw`
            -   Input this JSON: 
            ```py
            {
            "email": "<mcmaster_email_goes_here>",
            "password": "<password_goes_here>"
            }
            ```
        -   Send request
            -   If request is successful you will get a status 200!
            -   Take note of the cookies returned: 
                -   a session_token and csrftoken should be present under the `cookie` heading. These values are required for future APIs
* NOTE: Once a user has signed up, they can use the `POST /api/login/` directly, and skip the first two steps. 


### Running the Remaining API's  

#### Upload API 
Must run `POST /api/verify/` 
-   In Postman:
    -   set the API Method to `POST`
    -   set the URL to `http://127.0.0.1:8000/api/upload/`
    -   set the body type to `form-data`
        -  Add a key named `image` and ensure the type is file 
        -  Add the corresponding value to be the image that you want to upload. 
    - In the headers section, add these values: 
        -   key: `session_token`, value: <was_obtained_from_login_api>
        -   key: `X-CSRFToken`, value: <was_obtained_from_login_api>
    -   Send request
        -   If request is successful you will get a status 200!

### Alt-Text Generation Trigger 
Must run `POST /api/generate-alt-text/`
-   In Postman:
    -   set the API Method to `POST`
    -   set the URL to `http://127.0.0.1:8000/api/generate-alt-text/`
    -   set the body type to `form-data`
        -  Add a key named `image` and ensure the type is file 
        -  Add the corresponding value to be the image that you want to generate alt text for. 
    - In the headers section, add these values: 
        -   key: `session_token`, value: <was_obtained_from_login_api>
        -   key: `X-CSRFToken`, value: <was_obtained_from_login_api>
    -   Send request
        -   If request is successful you will get a status 200!
        -   The alt-text generated should be returned in the body. 

### Edit Generated Alt Text
Must run `PUT /api/edit-alt-text/`
-   In Postman:
    -   set the API Method to `PUT`
    -   set the URL to `http://127.0.0.1:8000/api/edit-alt-text/`
     -   set the body type to `raw`
            -   Input this JSON: 
            ```py
            {
            "entry_id": "<entry_id_goes_here>",
            "edited_alt_text": "<edited_alt_text_goes_here>"
            }
    - In the headers section, add these values: 
        -   key: `X-CSRFToken`, value: <was_obtained_from_login_api>
    -   Send request
        -   If request is successful you will get a status 200!

### Get History API 
Must run `GET /api/alt-text-history/`
-   In Postman:
    -   set the API Method to `GET`
    -   set the URL to `http://127.0.0.1:8000/api/alt-text-history/`
    - In the headers section, add these values: 
        -   key: `X-CSRFToken`, value: <was_obtained_from_login_api>
    -   Send request
        -   If request is successful you will get a status 200!
        -   The body outputted will have the images in base64 (can be converted as need) and their associated alt-text. 



 


