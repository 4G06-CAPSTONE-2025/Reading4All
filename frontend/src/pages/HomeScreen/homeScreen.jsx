import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import "./homeScreen.css";

export default function HomeScreen() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [history, setHistory] = useState([]);
    const [error, setError] = useState("");
    const [isDragging, setIsDragging] = useState(false);
    const [lastUploadedFile, setLastUploadedFile] = useState(null);
    const [altText, setAltText] = useState("");
    const [isGeneratingAltText, setIsGeneratingAltText] = useState(false);
    const [isHistoryOpen, setIsHistoryOpen] = useState(false);

    const fileInputRef = useRef(null);
    const navigate = useNavigate();

    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];

    const validateAndSetFile = (file) => {
        setError("");
        if (!file) {
            setSelectedFile(null);
            return;
        }
        if (!allowedTypes.includes(file.type)) {
            setError("Invalid file type. Please upload an image with extension JPEG, JPG, or PNG.");
            setSelectedFile(null);
            if (fileInputRef.current) {
                fileInputRef.current.value = null;
            }
            return;
        }
        setSelectedFile(file);
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        validateAndSetFile(file);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        validateAndSetFile(file);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDropzoneKeyDown = (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            if (fileInputRef.current) {
                fileInputRef.current.click();
            }
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        const item = {
            id: Date.now(),
            name: selectedFile.name,
            date: new Date().toLocaleString(),
            fileUrl: URL.createObjectURL(selectedFile),
            altText: null,
        };

        setHistory((prev) => [item, ...prev]);
        setLastUploadedFile(item);
        setAltText("");
        setSelectedFile(null);

        if (fileInputRef.current) {
            fileInputRef.current.value = null;
        }
    };

    const handleGenerateAltText = () => {
        if (!lastUploadedFile) return;

        setIsGeneratingAltText(true);
        setAltText("");

        const generated = `Example alternate text for "${lastUploadedFile.name}".`;

        const updated = {
            ...history[0],
            altText: generated,
        };

        setHistory((prev) => [updated, ...prev.slice(1)]);
        setLastUploadedFile(updated);
        setAltText(generated);
        setIsGeneratingAltText(false);
    };

    const handleToggleHistory = () => {
        setIsHistoryOpen((prev) => !prev);
    };

    const handleShowFullSessionHistory = () => {
        navigate("/session-history", { state: { history } });
    };

    const dropzoneAriaDescribedBy = error
        ? "upload-instructions upload-error"
        : "upload-instructions";

    return (
        <div className="home-container">
            {/* Logout button - top right */}
            <button
                type="button"
                className="logout-button"
                onClick={() => navigate("/")}
                aria-label="Log out and return to the login screen"
            >
                Logout
            </button>

            {/* Session history toggle (like ChatGPT sidebar toggle) */}
            <button
                type="button"
                className="history-toggle"
                onClick={handleToggleHistory}
                aria-pressed={isHistoryOpen}
                aria-label={isHistoryOpen ? "Hide session history panel" : "Show session history panel"}
            >
                {isHistoryOpen ? "Hide Session History" : "Show Session History"}
            </button>

            {isHistoryOpen && (
                <aside
                    className="home-sidebar"
                    aria-label="Session history for this session"
                >
                    <h2 className="sidebar-title">Session History</h2>
                    {history.length === 0 ? (
                        <p className="sidebar-empty">No images submitted yet.</p>
                    ) : (
                        <ul className="history-list">
                            {history.map((item) => (
                                <li className="history-item" key={item.id}>
                                    <span className="history-item-name">{item.name}</span>
                                    <span className="history-item-date">
                                        <span className="sr-only">Uploaded on </span>
                                        {item.date}
                                    </span>
                                </li>
                            ))}
                        </ul>
                    )}
                    <button
                        type="button"
                        className="history-full-button"
                        onClick={handleShowFullSessionHistory}
                        aria-label="Go to full session history page"
                    >
                        Show full session history
                    </button>
                </aside>
            )}

            <main className="home-main" role="main">
                <div className="home-main-inner">
                    <h1 className="home-title">Home Screen</h1>

                    <form
                        className="upload-form"
                        onSubmit={handleSubmit}
                        aria-label="Image upload form"
                    >
                        <div
                            className={`upload-dropzone ${isDragging ? "upload-dropzone--dragging" : ""}`}
                            tabIndex={0}
                            role="button"
                            aria-label="Upload image. Press Enter or Space to browse, or drag and drop a JPEG, JPG, or PNG file here."
                            aria-describedby={dropzoneAriaDescribedBy}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onKeyDown={handleDropzoneKeyDown}
                        >
                            <p className="upload-dropzone-main-text">
                                Drag and drop an image here
                            </p>
                            <p className="upload-dropzone-sub-text">
                                or use the button below
                            </p>
                            <button
                                type="button"
                                className="upload-choose-button"
                                onClick={() => fileInputRef.current && fileInputRef.current.click()}
                                aria-label="Browse files to choose an image"
                            >
                                Choose File
                            </button>
                        </div>

                        <input
                            id="image-upload"
                            ref={fileInputRef}
                            type="file"
                            accept="image/jpeg, image/jpg, image/png"
                            onChange={handleFileChange}
                            aria-label="Image file chooser"
                            aria-describedby={dropzoneAriaDescribedBy}
                            style={{
                                position: "absolute",
                                width: "1px",
                                height: "1px",
                                padding: 0,
                                margin: "-1px",
                                overflow: "hidden",
                                clip: "rect(0,0,0,0)",
                                border: 0,
                            }}
                        />

                        <p
                            id="upload-instructions"
                            className="upload-helper-text"
                        >
                            Acceptable formats: JPEG, JPG, PNG.
                        </p>

                        {error && (
                            <p
                                id="upload-error"
                                className="upload-error"
                                role="alert"
                                aria-live="polite"
                            >
                                {error}
                            </p>
                        )}

                        {selectedFile && (
                            <p className="selected-file">
                                Selected: <strong>{selectedFile.name}</strong>
                            </p>
                        )}

                        <button
                            type="submit"
                            className="upload-submit-button"
                            disabled={!selectedFile}
                            aria-disabled={!selectedFile}
                        >
                            Submit
                        </button>
                    </form>

                    {lastUploadedFile && (
                        <section
                            className="alt-text-section"
                            aria-labelledby="alt-text-title"
                            aria-describedby="alt-text-description"
                        >
                            <h2 id="alt-text-title" className="alt-text-title">
                                Alternate Text
                            </h2>
                            <p
                                id="alt-text-description"
                                className="alt-text-file-name"
                            >
                                Last uploaded file:{" "}
                                <strong>{lastUploadedFile.name}</strong>
                            </p>
                            <button
                                type="button"
                                className="alt-text-button"
                                onClick={handleGenerateAltText}
                                disabled={isGeneratingAltText}
                                aria-disabled={isGeneratingAltText}
                                aria-label="Generate alternate text for the last uploaded image"
                            >
                                {isGeneratingAltText
                                    ? "Generating..."
                                    : "Generate Alternate Text"}
                            </button>
                            <p
                                className="alt-text-output"
                                role="status"
                                aria-live="polite"
                            >
                                {altText}
                            </p>
                        </section>
                    )}
                </div>
            </main>
        </div>
    );
}
