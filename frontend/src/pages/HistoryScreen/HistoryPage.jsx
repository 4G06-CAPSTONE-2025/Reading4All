import React, { useState, useRef, use } from "react";
import { useNavigate } from "react-router-dom";
import "./HistoryPage.css";
import { useEffect } from "react";

export default function HistoryPage(){

    const navigate = useNavigate();

    const [history, setHistory] = useState([]);
    const [copiedAllHistory, setCopiedAllHistory] = useState(false);
    const [copiedEntryID, setCopiedEntryID] = useState(null); 
    const [error, setError] = useState("");

 
   const handleCopyAll = async () => {
        const allAltTexts = history.map(
            item => item.altText)
            .join('\n\n');

        try {
            await navigator.clipboard.writeText(allAltTexts);
            setCopiedAllHistory(true);
            setTimeout(() => {  
                setCopiedAllHistory(false);
            }, 1500);
        } 
        catch (err) {
            setError("Failed to copy to clipboard.");
        }
    };

    const handleCopyIndividualAltText = async (altText, id) => {
        try {
            await navigator.clipboard.writeText(altText);
            setCopiedEntryID(id);
            setTimeout(() => {
                setCopiedEntryID(null);
            }, 1500);
        } 
        catch (err) {
            setError("Failed to copy to clipboard.");
        }
    };

    useEffect(() => {
        fetch("http://127.0.0.1:8000/api/alt-text-history/")
        .then(response => response.json())
        .then(data => 
            {setHistory(data.history); })
        .catch(() => setError("Failed to get alt text history."));
        }, []);
        

    return (
        <div className="history-page">
            <div className="page-content"> 
            <div className="back-button-header">
            <button
            className="back-button" 
            onClick={() => navigate(-1)}>
                ← Back
            </button>
            </div>


            <div className="history-page-title">
                <h1>Alt Text History</h1>
            <p>
                View and copy previously generated alternative text for your 10 most recent uploaded images, 
                with the most recent shown first.
            </p>
            <button 
            className="copy-all-button"
            onClick={handleCopyAll}
           >
                {copiedAllHistory ? "✓ Copied" : "Copy All Alt Texts"}
            </button>            
            </div>

            {/* if unable to copy all text */}
            {
                error && (
                <p className="error-text">
                {error}
                </p>
            )}


            <div className="history-list">
                {history.length === 0 ? (
                    <p>No history available.</p>
                ) : (
                    history.map((item, index) => (
                        <div key={index} className="history-item">
                            <div className="history-card-left-col">
                            <img 
                            src={ `data:image/png;base64,${item.image}`} 
                            alt="Uploaded image preview"
                            className="history-image"
                            />
                            </div>                            

                            <div className="history-card-right-col">
                                <div className="alt-text-label">
                                <p className="alt-text">
                                    {item.altText}
                                </p>
                                </div>
                                <button className="copy-individual-text-button"
                                onClick={() => handleCopyIndividualAltText(item.altText, index)}
                                >
                                {copiedEntryID === index? "✓ Copied" : "Copy Alt Text"}

                            </button>

                       

                            </div>

                        </div>
                
                ))
                )}
            </div>
            </div>


        </div>
    );
}

