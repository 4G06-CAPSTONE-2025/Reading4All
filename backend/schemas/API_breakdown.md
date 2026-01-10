## POST /api/upload:
-	This API receives the user’s diagram image, validates the image using the file type and size. Finally, the API will stores the image and return a status code to reflect if it was successfully uploaded or a failure occurred.
#### Request Data Schema
```json
{
  "type" : "object",
  "required" : [ "image", "user_id"],
  "properties" : {
    "image" : {
      "type" : "string"
    },
    "user_id" : {
      "type" : "string"
    }
  }
}
```

#### Success Response Data Schema
```json
{
    "type" : "object",
  "required" : ["status"],
  "properties" : {
    "image_id" : {
      "type" : "string"
    },
    "status" : {
      "type" : "string",
      "enum": ["SUCCESS", "ERROR"]
    },
    "error":{
        "type": "string",
        "enum":["INVALID_FILE_TYPE", "FILE_SIZE_INVALID", "UNAUTHORIZED_ACCESS"]

    }
  }
}
```

##	POST /api/alt-text/generate:
-   This API prompts the generation of alt-text of the successfully uploaded image and returns the generated alt-text or an error status. 

#### Request Data Schema 
```json
{
  "type" : "object",
  "required" : [ "image_id"],
  "properties" : {
    "image_id" : {
      "type" : "string",
    }
  }
}
```


#### Response Data Schema
```json
{
  "type" : "object",
  "required" : [ "status"],
  "properties" : {
    "status" : {
      "type" : "string",
       "enum": ["SUCCESS", "ERROR"]
    },
    "alt_text":{
        "type":"string"
    },
    "error":{
        "type": "string",
        "enum":["GENERATION_TIMEOUT"]
    }
  }
}
```

## PUT /api/alt-text/edit 
-   This API updates the alt-text associated with an uploaded image based on user changes.

#### Request Data Schema 
```json
{
  "type" : "object",
  "required" : [ "image_id", "edited_alt_text"],
  "properties" : {
    "image_id" : {
      "type" : "string",
    },
     "edited_alt_text" : {
      "type" : "string",
    }
  }
} 
```

#### Response Data Schema 
```json
{
  "type" : "object",
  "required" : [ "status"],
  "properties" : {
    "status" : {
      "type" : "string",
       "enum": ["SUCCESS", "ERROR"]
    },
    "alt_text":{
        "type":"string"
    },
    "error":{
        "type": "string",
        "enum":["UNABLE_TO_SAVE"]
    }
  }
}