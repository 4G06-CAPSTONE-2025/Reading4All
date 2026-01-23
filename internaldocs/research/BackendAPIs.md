

## PUT /api/alt-text/edit

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
- image_id: a string that uniquely identifies the uploaded image whose alt-text is being edited.
- edited_alt_text: a string representing the updated alt-text provided by the user.

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