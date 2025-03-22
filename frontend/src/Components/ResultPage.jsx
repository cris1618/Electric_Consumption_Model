import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "../Styles/ResultsPage.css"; 
import Atene from "../Styles/Images/romaguidetour-visite-guidate-personalizzate-roma-leonardo-vinci-scuola-atena-square.jpg";

const ResultPage = () => {
  const navigate = useNavigate();
  const { state } = useLocation();
  const { 
    prediction,
    month,
    size,
    occupants,
    heating_type,
    cooling_type,
   } = state || {};

   const [suggestion, setSuggestion] = useState("Loading suggestions...");

   
      useEffect(() => {
        const prompt = `
          The user has a monthly energy consumption of ${prediction} KWh.
          Additional household information:
          - Month: ${month}
          - House Size: ${size} sq ft
          - Number of Occupants: ${occupants}
          - Heating Type: ${heating_type}
          - Cooling Type: ${cooling_type}
        
          Provide practical and actionable suggestions to reduce energy consumption and save money.
          Keep it short. Within 100 words.
        `;
      
        fetch("http://127.0.0.1:8000/api/suggestions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }) 
        })
          .then(res => {
            if (!res.ok) {
              throw new Error(`HTTP error! Status: ${res.status}`);
            }
            return res.json();
          })
          .then(data => {
            console.log("Suggestion:", data);
            setSuggestion(data.suggestion);
          })
          .catch(err => {
            console.error("API error:", err);
            setSuggestion("No suggestions available due to an error.");
          });
      }, [prediction, month, size, occupants, heating_type, cooling_type]);
       
        return (
          <div className="result-page"
          style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,0.2), rgba(255,255,255,0.7)), url(${Atene})`,
          }}>
            <div className="result-container">
              <h1>Your Predicted Monthly Energy Consumption is:</h1>
              <h2>{prediction} KWh</h2>
              <h3>Your Suggestions</h3>
              <div className="suggestions-output">
                {suggestion ? <p>{suggestion}</p> : <p>No suggestion available.</p>}
              </div>
              <button 
                className="brown-button"
                onClick={() => navigate("/")}
              >
                Start Over
              </button>
            </div>
          </div>
        );
      }
      
      export default ResultPage;
