## API: PUT /api/alt-text/edit

### Purpose
- This endpoint allows a user to update previously generated alt-text for an uploaded image. It ensures that:
    - The edited alt-text is saved especially to update show history
    - The edited alt-text remains associated with the correct image

### HTTP Methods: PUT 
- Since the alt-text already exists (from POST /api/alt-text/generate), editing it is an update which makes it a PUT method. 
- In RESTful API design:
    - ```POST``` → creates new resources
    - ```GET``` → retrieves resources
    - ```PUT``` → updates an existing resource
    - ``DELETE``→ removes a resource

### Status Codes 
- Status codes are three-digit HTTP codes returned by a server in response to a client's request. 
- They show the outcome of the request, such as whether it was successful, if an error occurred, or if further action is needed. 
- These codes help especially for debugging and managing communication between software systems. 
- Status codes (specifically for this API):
- ```200 OK``` - Alt-text successfully updated
- ```400 Bad Request``` - Invalid or missing request fields
- ```403 Forbidden``` - User does not own the image
- ``404 Not Found`` - Image or alt-text record does not exist 
- ``500 Internal Server Error`` - Alt-text could not be saved

### Request Data Schema (Shown in /backend/schemas_docs/API_breakdown.md)
- ``image_id``: a string that uniquely identifies the uploaded image whose alt-text is being edited.
- ``edited_alt_text``: a string representing the updated alt-text provided by the user.

### Response Data Schema (Shown in /backend/schemas_docs/API_breakdown.md)
- Response follows this schema:
```json
{
  "alt_text": "string",
  "error": "UNABLE_TO_SAVE"
}
```
- On success, the API returns the updated alt-text:
```json
{
  "alt_text": "Modified description of the diagram"
}
```
- If the alt-text cannot be saved due to failure, the API returns:
```json
{
  "error": "UNABLE_TO_SAVE"
}
```

## Using Django for API
### What is Django
- Django is a high-level Python web framework that supports rapid, secure backend development. When combined with Django REST Framework (DRF), Django provides built-in tools for implementing RESTful APIs. This includes request validation, authentication, database integration, and structured responses.

### URL Routing
- Django uses URL routing to map HTTP requests to backend logic.
    - Each API endpoint is registered in a ``urls.py`` file (shown in /backend/api/urls.py).
    - The HTTP method (``PUT``) determines which logic is executed.
    - Clear routing keeps API endpoints modular and readable.
- Example responsibility:
    - Route ``/api/alt-text/edit`` to a dedicated view handling alt-text updates.

### Views
- In Django REST Framework (DRF), views handle incoming API requests.
- For this endpoint, the view is responsible for:
    - Accepting a ``PUT`` request
    - Extracting ``image_id`` and ``edited_alt_text``
    - Performing validation and permission checks
    - Updating the database
    - Returning a structured response
- DRF supports:
    - Function-based views (simpler)
    - Class-based views (more scalable and reusable)
    - For early development and clarity, function-based views are often easier to reason about.

### HTTP Status Codes & Responses
- Django REST Framework standardizes API responses by:
    - Returning JSON responses
    - Associating them with appropriate HTTP status codes
- For this endpoint:
    - ``200 OK`` indicates a successful update
    - Error responses indicate validation, permission, or system failures
    - Returning structured responses ensures consistent frontend behavior.

### Separation of Concerns
- Django enforces a clear separation between:
    - URL routing (where the request goes)
    - Views (what the request does)
    - Models (how data is stored) - *our system will be using Supabase (next section)*
- This separation improves:
    - Maintainability
    - Testability
    - Long-term scalability

## Supabase for Database Layer

### Supabase 
- Supabase is an open-source backend platform designed to simplify application development by providing managed backend services
- It offers a PostgreSQL database, authentication, file storage, and API access through a unified platform. 
- Supabase is commonly used as an alternative to traditional backend-as-a-service solutions while retaining the flexibility of a relational database.

### PostgreSQL-Based Database
- Supabase is built on top of PostgreSQL, a widely used relational database management system. 
- This allows developers to take advantage of structured schemas, relational constraints, and transactional guarantees. Because PostgreSQL is fully SQL-compliant, Supabase supports complex queries, joins, and indexing, which makes it suitable for applications that manage structured and interrelated data.

### Authentication and Authorization
- Supabase provides built-in authentication services that support email/password login and token-based authentication. 
- It also includes row-level security (RLS) policies, which allow developers to define fine-grained access controls directly at the database level. These features help ensure that users can only access or modify data they are authorized to interact with.

### Storage and Data Management
- In addition to structured data storage, Supabase offers object storage for handling files such as images and documents. 
- This enables applications to manage both structured and unstructured data within the same backend ecosystem. 
- Supabase also provides automatic data persistence and backup features, reducing the operational overhead required to manage storage infrastructure.

### Supabase for Reading4All System
- In the Reading4All system, Supabase is used as the primary data storage solution for images, generated alt-text, and alt-text edit history. Its PostgreSQL foundation allows structured relational data to be stored efficiently while supporting complex queries and constraints.

## Django and Supabase Integration
- In the Reading4All backend subsystem, Django serves as the API and application logic layer, while Supabase functions as the persistent data store. 
- Django is responsible for handling HTTP requests, validating input data, enforcing authentication and authorization rules, and coordinating interactions with the database. 
- Supabase is responsible for storing and managing the system’s data, including alt-text records and edit history.
- For the ``PUT`` API:
    - When a user edits alt-text, the Django API processes the PUT /api/alt-text/edit request
    - Upon successful validation, writes the updated alt-text to the appropriate table in Supabase. 
    - This layered architecture separates business logic from data storage, improving maintainability, security, and scalability of the Reading4All system.


## Reason for research 
- Minimal experience in backend systems and implementing APIs. 
- Need to implement the PUT /api/alt-text/edit but want to learn more about backend first. 