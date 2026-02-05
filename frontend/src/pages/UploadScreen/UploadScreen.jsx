import React, { useState, useRef, useEffect } from "react";
import "./UploadScreen.css";

export default function HomeScreen(){

    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewImg, setPreviewImg] = useState(null)
    const [error, setError] = useState(""); 

    const [altText, setAltText] = useState("")

    const [copiedAltText, setCopiedAltText] = useState(false);


    // for the screen reader to announce to user success alerts
    const successMessageRef = useRef(null); 

    // for the screen reader to announce to user error alerts
    const errorMessageRef = useRef(null);

    const hasAltText = altText && altText.trim().length > 0;

    const fileInputRef = useRef(null);
    
    const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        setIsDragging(false);
    };


    const handleImageValidationBackend = async (image) => {
        const formData = new FormData();
        formData.append("image", image);

        const response = await fetch("https://reading4all-backend.onrender.com/api/upload/",
            {
                method: "POST",
                body:formData,
                credentials: "include"
            }
        )
        if (response.ok){
            return true
        }
        const msg= await response.json();
        throw new Error(msg.error);
    }

    const handleUploadError = (errorMssg) => {
        if (errorMssg === "MISSING_IMAGE")
        {
            setError("No image was uploaded. Please select an image")
        }
        else if (errorMssg === "INVALID_FILE_TYPE")
        {
            setError("Invalid File Type. Please upload a PNG, JPEG or JPG image")
        }
        else if (errorMssg === "FILE_SIZE_INVALID")
        {
            setError("Image size exceeds 10 Megabytes.")
        }
        else if (errorMssg === "UNAUTHORIZED_ACCESS_OR_CORRUPTED")
        {
            setError("Image file is corrupted or cannot be opened")
        }
        else
        {
            setError("Image validation failed. Please try again")
        }
    }

    const handleImageDrop = async (e) =>{

        e.preventDefault();
        setIsDragging(false)

        const image_uploaded = e.dataTransfer.files[0]

        if (!image_uploaded) return;

        try 
        {
            await handleImageValidationBackend(image_uploaded)
        }
        catch (err)
        {
            handleUploadError(err.message)
            return
        }
        
        const url = URL.createObjectURL(image_uploaded)
        setError("");
        setSelectedFile(image_uploaded);
        setPreviewImg(url);
        resetAltTextGenProcess();
    };

    const handleFileSelect = async (e) => {
        const image_uploaded = e.target.files[0]
        
        if (!image_uploaded) return;

        try
        {
            await handleImageValidationBackend(image_uploaded)
        }
        catch (err)
        {
            handleUploadError(err.message)
            return
        }

        const url = URL.createObjectURL(image_uploaded)
        setError("");
        setSelectedFile(image_uploaded);
        setPreviewImg(url);
        resetAltTextGenProcess();
    };

    const handleRemoveImg = (e) => {
        setError("")
        setSelectedFile(null);
        setPreviewImg(null);
        resetAltTextGenProcess();
    }
     

    const handleImageDropBox = (e) =>
    {
         if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            if (fileInputRef.current) {
                fileInputRef.current.click();
            }
        }
    };

    const resetAltTextGenProcess = () => {
        setAltText("");
    }

    const handleCopyAltText= async () => {
        try {
            await navigator.clipboard.writeText(altText);
            setCopiedAltText(true);
            setTimeout(() => {
                setCopiedAltText(false);
            }, 1500);
        } 
        catch (err) {
            setError("Failed to copy to clipboard.");
        }
    };

    const handleGenerateAltText = () =>{
        const formData = new FormData();
        formData.append("image", selectedFile);

        fetch("https://reading4all-backend.onrender.com/api/generate-alt-text/",
            {
                method: "POST",
                body:formData,
                credentials: "include"
            }
        )
        .then(response => {
            if (response.ok)
            {
                return response.json()
            }
            throw new Error("Unable to get alt text history") 
        })
        .then(data => {
            setAltText(data.alt_text);
            setError("");
        })
        .catch(() =>{
            setError("Failed to Generate Alt Text. Please try again!");
        });
    };

     useEffect(() => {
        if (error && errorMessageRef.current) {
            errorMessageRef.current.focus();
            return
        }
        if ((selectedFile || hasAltText) && successMessageRef.current) {
            successMessageRef.current.focus();
        }
    }, [error, selectedFile, hasAltText]);


    return (
    <div className="upload-page">

         <div className="title-section">

        <header className = "upload-page-title">
        <h1>Alternative Text Generation</h1>

        <p>
        Generate clear, concise alternative text for STEM diagrams 
        </p>
        </header>
        
        </div>
        
        <div className="upload-page-content">
        <div className="upload-section">
        <div className={`upload-drag-file-box ${isDragging ? "upload-frame-dragging" : ""}`}
        tabIndex={0}
        aria-label="Upload image. Press Enter or Space to browse, or drag and drop a JPEG, JPG, or PNG file here."
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleImageDrop}
        onKeyDown={handleImageDropBox}
        >
            <div className="upload-frame-content">

            <p className="upload-frame-title"> 
                Drag and Drop an Image Here
            </p>

            <p className="upload-frame-select-img-title">
                or upload the file using the button below
            </p>

            <input
				type="file"
                ref={fileInputRef}
				accept="image/png, image/jpeg, image/jpg"
				onChange={handleFileSelect}
                className="hiding-classic-button"
                aria-label="Upload image for alt text generation"
			/>

            <button
            className="upload-button"
            onClick={()=>fileInputRef.current?.click()}
            >
				Choose File
			</button>

            <p className="upload-info-text">
                Supported formats: JPG, PNG or JPEG files up to 10 Megabytes
            </p>
            </div>

           </div>
            </div>

        </div>

         { selectedFile && (
            <div className="upload-status-encloses">
                <div className="upload-success-view">
                <p className="success-title"
                ref={successMessageRef}
                tabIndex="-1"
                >
                    {
                        hasAltText? "Successfully Generated Alt Text!" :
                        "Successfully Uploaded Image!"
                    }                   
                </p>


                <div className="image-del-sec">
                <div className="upload-view-of-img">
                <img src={previewImg} alt="Uploaded preview" className="img-preview"/>
                 <button 
                    className="delete-button" 
                    onClick={handleRemoveImg}
                    aria-label="Remove uploaded image"
                >
                    Remove
                </button>
                 </div>
                <div className="file-name-gen-button">
                {!hasAltText ? (
                     <p className="file-name-text"
                >
                    {selectedFile.name}
                    
                </p> ): ""}
               

                {!hasAltText ? (
                    <button className="gen-alt-text-button"
                        onClick={handleGenerateAltText}
                    >
                    Generate Alt Text
                    </button>

                ): ""}
                
                {hasAltText ? (

                 <textarea
                    className="computed-alt-text-box"
                    onChange = {(e) => setAltText(e.target.value)}
                    value={altText}
                    rows={16}
                    cols={70}
                    aria-label="Generated alt text for the uploaded image that can be edited"
                />
                
                    
                ) : "" }
               
               <div className= "save-changes-copy-alt-text-buttons">
                
                {hasAltText ? (
                    <button 
                    className="save-edits-button"
                    >
                    Save Edits
                    </button>
                ) : "" }

                {hasAltText ? (
                    <button className="copy-individual-text-button-upload-pg"
                        onClick={handleCopyAltText}
                     >
                    {copiedAltText? "✓ Copied" : "Copy Alt Text"}
                    </button>
                ): ""}
                </div>


                </div>
       
                </div>
                    </div>

                    
            </div>
        ) }
        

    { error && !selectedFile && (
            <div className="upload-status-encloses">
                <div className="upload-error-view"
                    role="alert"
                    aria-live="assertive"
                >
                <div className="error-content">
                    <div className="error-logo"> 
                        !
                    </div>
                    <div className="error-text">
                    <p className="error-title">
                        Upload Failed
                    </p>
                    <p className="error-message"
                        ref={errorMessageRef}
                        tabIndex="-1"
                    >
                        {error}
                    </p>

                   </div>
            </div>
        </div>
    </div>
)}
    </div>
    );
}
