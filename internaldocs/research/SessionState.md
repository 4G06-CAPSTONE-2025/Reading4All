### Web Session:
-    A sequence of network HTTP request and response transactions associated with the same user. 
Web applications need the information from a user spanning across multiple requests and sessions allow you to do so. 
-   Web applications make use of sessions once the user has been authenticated, ensuring that subsequent requests of the users can be identified. 
-   Once an authenticated session has began, the session ID (or Token) is strong authentication method. The session ID binds the authenticated user with their HTTP requests. 
    #### Session ID Properties
    -   Session ID is assigned a session creation time and is a name=value pair that is shared on every HTTP request during the session. 
    ### Built-in Session Management Implementations 
    -   Web development frameworks like .NET have their own session management features and associated implementation. However, React does not have a built in functionality. 
    #### Session Expiration: 
    -   Session Expiration plays an important role in minimizing attacks on the application. The longer a valid session is  the longer an attacker can launch an attack over active sessions to hijack them.
    -   Therefore, the session expiration timeout must consider the functionality of the web application as well as balance security. 
    -   When a session expires, the web application must invalidate the session on both the client and server side. 
        -   The client side must stop sending the session ID and server side must clear the ID. 
    ##### Types of Session Expiration: 
    1) Automatic Session Expiry:  This is when a session expires after the user has be inactivate for a period of time. Timeout is measured from the last HTTP request.
    2) Absolute Timeout: Defines the maximum lifetime of a session, regardless of activity. 
    3) Renewal Timeout: In the middle of the user session, the web application can generate a new valid session ID. 

### Ideas for implementing our 2 hour Session
-   The session should be created at login, when the login API is invoked. 
    -   A session ID can be made here. 
        -   FastAPI has .set_cookie function which can be used like this: 
        ```py
            response.set_cookie(key="fakesession", value="fake-cookie-session-value")
        ```
        - Store the session ID and when it expires in a data type, for example a dict.
-   At every request the session should be validated. This can be done by calling a function which validates the session ID. 