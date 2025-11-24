import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./showHistory.css";

export default function ShowHistory() {

    const navigate = useNavigate();
    return (
        <div className="history-container">
             <button
            type="button"
            className="back-button"
            onClick={() => navigate("/main")}
            aria-label="Return to the main screen"
            >
                Back to Home Page
            </button>
        </div>

  );
}