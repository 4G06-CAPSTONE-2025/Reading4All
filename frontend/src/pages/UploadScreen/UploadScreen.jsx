import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./UploadScreen.css";

export default function HomeScreen(){

    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewImg, setPreviewImg] = useState(null)
    const [error, setError] = useState(""); 

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

        setError("")
        setSelectedFile(image_uploaded)
        setPreviewImg(url)
    };

    const handleFileSelect = (e) => {
        const image_uploaded = e.target.files[0]
        const url = URL.createObjectURL(image_uploaded)
        
        if (!image_uploaded) return;
        setError("");
        setSelectedFile(image_uploaded)
        setPreviewImg(url)
    };

    const handleRemoveImg = (e) => {
        setSelectedFile(null);
        setPreviewImg(null);
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
        
        <div className="page-content">

        <div className={`upload-frame ${isDragging ? "upload-frame-dragging" : ""}`}
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

         { selectedFile && (
            <div className="upload-success-encloses">
                <div className="upload-success-view">
                <p className="success-title">                        
                Successfully Uploaded Image!
                </p>
                <div className="image-del-sec">
                <div className="upload-view-of-img">
                <img src={previewImg} alt="Uploaded preview" className="img-preview"/>
                 <button className="delete-button" onClick={handleRemoveImg}>
                    Remove
                </button>
                 </div>
                <div className="file-name-gen-button">

                <p className="file-name-text">{selectedFile.name}</p>

                <button className="gen-alt-text-button">
                    Generate Alt Text
                </button>

                </div>

                </div>
                    </div>

                    
            </div>
        ) }
        

    { error && (
            <p className="upload-error-mssg">
                Upload Failed
            </p>
        )  }
    </div>


)

}


