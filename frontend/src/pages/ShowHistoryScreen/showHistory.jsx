import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./showHistory.css";

export default function ShowHistory() {

    const navigate = useNavigate();
    const location = useLocation();

    const history = location.state?.history || [];

    const [selectedItem, setSelectedItem] = useState(null);

    return (
        <div className="history-page">
            <button
                className="back-button"
                onClick={() => navigate("/main")}
                aria-label="Return to the main screen to upload images"
            >
                Back to Home
            </button>

            <h1 className="history-title">Full Session History</h1>

            {history.length === 0 ? (
                <p style={{marginLeft: 30}}>No Previously Uploaded Images</p>
            ) : (
                <ul className="history-list" aria-label="Uploaded file history">
                    {history.map((item) => (
                        <li
                            key={item.id}
                            className="history-entry"
                            role="button"
                            tabIndex={0}
                            onClick={() => setSelectedItem(item)}
                            onKeyDown={(e) =>
                                e.key === "Enter" && setSelectedItem(item)
                            }
                            aria-label={`View preview for ${item.name}`}
                        >
                            <p><strong>File:</strong> {item.name}</p>
                            <p><strong>Uploaded:</strong> {item.date}</p>
                            <p>
                                <strong>Alt Text:</strong>{" "}
                                {item.altText || "No alt text generated yet."}
                            </p>
                        </li>
                    ))}
                </ul>
            )}

            {/* image preview when item is clicked */}
            {selectedItem && (
                <div className="history-preview">
                    <h2>{selectedItem.name}</h2>
                    <img
                        src={selectedItem.fileUrl}
                        alt={selectedItem.altText || selectedItem.name}
                        className="preview-image"
                    />
                    <p className="preview-alt">
                        <strong>Alt Text:</strong> {selectedItem.altText}
                    </p>
                </div>
            )}
        </div>
    );
}