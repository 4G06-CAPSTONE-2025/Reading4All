import React, { useState } from "react";
import "./HistoryPage.css";
import { useEffect } from "react";

export default function HistoryPage(){

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
        fetch("https://reading4all-backend.onrender.com/api/alt-text-history/",
            {
                method: "GET",
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
            setHistory(data.history);
            setError("");
        })
        .catch(() =>{
            setError("Failed to Load History. Please try again!");
        });
        }, []);
        

    return (
        <div className="history-page">
            <div className="history-page-content"> 

            <div className="history-page-title">
                <h1>Alt Text History</h1>
            <p>
                View and copy previously generated alternative text for your 10 most recent uploaded images, 
                with the most recent shown first.
            </p>
            { history.length> 0 && !error && (
            <button 
            className="copy-all-button"
            onClick={handleCopyAll}
           >
                {copiedAllHistory ? "✓ Copied" : "Copy All Alt Texts"}
            </button>   
            )}         
            </div>
          
            {/* if unable to copy all text or load history data */}
            {error && (
            <div className="error-case-box"
            role="alert"
            >

                    <p className="error-symbol">
                           ⚠
                    </p>

                    <p className="error-text">
                        {error}
                     </p>
            </div>
            )}


            <div className="history-list">
                {(history.length === 0 && !error) ? (
                    <p>No alternative text has been generated yet!</p>
                ) : (
                    history.map((item, index) => (
                        <div key={index} className="history-item">
                            <div className="history-card-left-col">
                            <img 
                            src={ `data:image/png;base64,${item.image}`} 
                            alt=""
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
                                aria-label = {`Copy alternative text: ${item.altText}`}
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

