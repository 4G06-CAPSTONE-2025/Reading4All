# Reading4All -  Frontend Overview and Schedule   
### Jan 5th to Jan 12th:  Figma Designs 
Goal: Finalize the different pages to be made and work on Figma designs for each page. Ensure we keep in mind the Look and Feel Requirements and Accessibility requirements that were previously defined
-   Work can be split up by pages after the list of pages has been finalized
Pages:
-   Make sure everything is spaced out 
-	Home page. 
    -   Has an introduction
    -   Has connection to login and sign-up page
-	Login page
    -   Username and password field 
    -   Sign in button 
    -   Error message for failed login 
    -   Sign up page
    -   First Name, Last Name, email, password and confirm password
    -   Error message if passwords don’t match or email is invalid format 
-	Main upload image page
    -   File explore that only lets you select images 
    -   Generate alt button 
    -   Show history button 
    -   Sign out button 
    -   If they hover over the alt text, the program should automatically copy. 
    -   Should also include a copy button  
    -   Error Messages:
        -   Alt text generation time out 
        -   No image uploaded 
        -   Unable to copy 
-	Show history page 
    -   Includes last 10 images 
    -   Copy button -> make sure this accessible 
    -   Back button that takes you to main upload image page 
    -   Sign out? Is this more accessible?
-	Session expired error message

Outcome: 
-	Written outline for each page and its main functionality
-	Figma design for each page 
-	Figma designs for error messages and failure cases. For example, when a user inputs invalid login, when AI model times out, etc. 
-	Do we want to make the Figma interactive to make sure user flow makes sense?? 
    -   Yes
-	Setup meeting with Team to review design and make issues for needed changes
    -   Ensure that checklist made in V&V is followed for this review. 
-	Setup meeting with Jing to review preliminary design and ensure we are on track to meet WCAG and AODA standards.  
### Jan12th – Jan 21:  Implement new pages and Test screen reader compatibility 
Goal: Add new pages to frontend, as well as modify existing ones to match Figma design. Ensure we test screen reader compatibility with pages.  
Outcome: 
-	Frontend for all pages, which are screen reader compatible and keyboard-only navigation.
-	Git Issues to fix anything 
### Jan 21 – Jan 30: Integration with Backend 
Goal: connect frontend pages to backend APIS to ensure end-to-end system behavior 
Outcome: 
-	Fully integrated frontend and backend 
-	Verified end-to-end user flow
Implement error messages
Goal: Ensure error popups are implemented using Figma design and follow accessibility requirements. 
Outcome: 
-	All error paths have been considered and implemented in the Frontend  
Automated WCAG Testing 
Goal: Work with Jing to complete automated WCAG testing. Create GIT issues to fix any problems that arise. 
Outcome: 
-	WCAG automated testing approval 


