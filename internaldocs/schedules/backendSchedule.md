# Reading4All - Backend Overview and Schedule  
## API List 
### 1.	POST /api/upload:
-	Receives the user’s diagram image, validates the image using the file type and size. Finally, the API will stores the image and return a status code to reflect if it was successfully uploaded or a failure occurred.
### 2.	POST /api/alt-text/generate:
-	This is a post request as the system is computing a new alt-text for the specific image. The uploaded image is included in the request.
-	This will return the generated alt text. 
### 3.	PUT /api/alt-text/edit 
-	Saves any changes the user makes to the generated alt-text.  Ensures that the image uploaded by the user is now associated with their edited alt-text. 
-	Returns a status code indicating success or failure 
### 4.	GET /api/alt-text
-	Uses session token or auth token to retrieve the recent uploaded images and their generated alt-text. 
### 5.	POST /api/auth/login 
-	Authenticates the user and starts a session
-	Returns status code. 
### 6.	POST/api/auth/logout
-	Ensures all user data is deleted and not stored, as well as ends the user’s session.  

## Backend Development Schedule  
### Week 1 & 2 -> API Planning and Design 
Goal: Ensure team agrees on APIS needed. 

Tasks: 
-	Finalize list of APIS needed 
-	Discuss the request and response formats and write up schemas.
-	Finalize how session and authorization will be done 
-	Determine how long a session will be 
-	API flow diagram to show how they connect 
-	Finalize technology that will be used and ensure setup 

Outcome: 
-	API list document 
-	Schema document for each API to outline expected responses and request formats
o	Need to ensure frontend and AI is aligned 
-	Diagram showing API connections and how they relate 
-	Decision on session period
o	Do research on how session states work if more info is needed.
-	Finalize backend stack
    -    Languages and framework
    - Testing and coverage frameworks 
    - What databases will be used and their functionality
-	Reading4All Repo has initial setup to begin implementing
    - This includes the coverage and unit testing frameworks that will be used for unit testing later on

NOTE: After Week 1 and 2, Backend Team should be ready for implementation. 
### Week 3 & 4.5 -> API Implementation 
Goal: Implement finalized APIs and ensure they can be run locally. Complete manual basic testing. 
-	Can split up work by APIs so work can be done concurrently. 

Outcome: 
-	All APIs that have been finalized from previous are implemented 
-	Manual testing is completed for each API
o	Refer to determined tests in V&V Plan
-	Issues made for any missing gaps in implementation 
o	Must fix these issues prior to continuing 

### Week 5 -> API Testing 
Goal: Write unit tests and ensure all error cases are handled correctly. 
-  These come from V&V plan 
- Ensure that we meet specified code coverage percentage and modify if needed
- Tests must cover any error paths to ensure user experience. For example: inputting invalid image types, not receiving input from AI module. 

Outcome: 
-	Code coverage percentage has been met by unit tests written 
-	Team must be confident in the behavior of the APIs in isolation. 
-	Git Workflow should be modified to run unit tests when new code is merged. This will ensure that existing correct behavior is not accidentally changed. 
### Week 6 & 7 -> API Integration  
Goal:  Ensure API’s work together and data is stored correctly. This includes setting up database if needed or researching how to store in a session state. 
-	Integrate backed APIs with the front end 

Outcome: 
-	Confirm backend behaves correctly end-to-end




