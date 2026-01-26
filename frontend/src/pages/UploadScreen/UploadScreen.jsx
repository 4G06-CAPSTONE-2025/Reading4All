import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./UploadScreen.css";

export default function HomeScreen(){

    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewImg, setPreviewImg] = useState(null)
    const [error, setError] = useState(""); 

    const [altText, setAltText] = useState("")

    const mockAltText = `
    Far far away, behind the word mountains, far from the countries Vokalia and Consonantia,
    there live the blind texts. Separated they live in Bookmarksgrove right at the coast of
     the Semantics, a large language ocean. A small river named Duden flows by their place
      and supplies it with the necessary regelialia. It is a paradisematic country, in which 
      roasted parts of sentences fly into your mouth. Even the all-powerful Pointing has no 
      control about the blind texts it is an almost unorthographic life One day however a 
      small line of blind text by the name of Lorem Ipsum decided to leave for the far World
       of Grammar. The Big Oxmox advised her not to do so, because there were thousands
    `

    const hasAltText = altText && altText.trim().length > 0;

    const fileInputRef = useRef(null);
    
    const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        setIsDragging(false);
    };

    const handleImageDrop = (e) =>{

        e.preventDefault();
        setIsDragging(false)

        const image_uploaded = e.dataTransfer.files[0]
        const url = URL.createObjectURL(image_uploaded)

        if (!image_uploaded) return;
        if (!validateFile(image_uploaded)) return;


        setError("");
        setSelectedFile(image_uploaded);
        setPreviewImg(url);
        resetAltTextGenProcess();
    };

    const handleFileSelect = (e) => {
        const image_uploaded = e.target.files[0]
        const url = URL.createObjectURL(image_uploaded)
        
        if (!image_uploaded) return;
        if (!validateFile(image_uploaded)) return;

        setError("");
        setSelectedFile(image_uploaded);
        setPreviewImg(url);
        resetAltTextGenProcess();
    };

    const handleRemoveImg = (e) => {
        setSelectedFile(null);
        setPreviewImg(null);
        resetAltTextGenProcess();
    }
        console.log("altText:", altText);
        console.log("hasAltText:", hasAltText);

    const handleImageDropBox = (e) =>
    {
         if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            if (fileInputRef.current) {
                fileInputRef.current.click();
            }
        }
    };

    const MAX_FILE_SIZE = 10485760;

    const validateFile = (file) => {
        if(!file) return false

        if(file.size>MAX_FILE_SIZE)
        {
        setError("File size exceeds 10 Megabytes");
        setSelectedFile(null);
        setPreviewImg(null);
        return false;
        }
        setError("");
        return true;
    }

    const resetAltTextGenProcess = () => {
        setAltText("");
    }


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
        <div className={`upload-drag-file-box${isDragging ? "upload-frame-dragging" : ""}`}
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
                <p className="success-title">     
                    {
                        hasAltText? "Successfully Generated Alt Text!" :
                        "Successfully Uploaded Image!"
                    }                   
                </p>
                <div className="image-del-sec">
                <div className="upload-view-of-img">
                <img src={previewImg} alt="Uploaded preview" className="img-preview"/>
                 <button className="delete-button" onClick={handleRemoveImg}>
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
                        onClick={() => setAltText(mockAltText)}
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
                />
                
                    
                ) : "" }
               
                
                {hasAltText ? (
                    <button 
                    className="save-edits-button"
                    hidden={!hasAltText}
                    >
                    Save Edits
                    </button>
                ) : "" }

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
                    <p className="error-message">
                       File size exceeds 10 Megabytes. Please Try again!   
                    </p>

                   </div>
            </div>
        </div>
    </div>
)}
    </div>
    );
}
