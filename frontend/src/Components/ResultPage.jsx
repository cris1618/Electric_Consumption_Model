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

   const [suggestion] = useState("Loading suggestions...");

   useEffect(() => {
    const prompt = `
    //The user has a monthly energy consumption of ${prediction} KWh.
    //  Additional household information:
     //- Month: ${month}
     //- House Size: ${size} sq ft
     //- Number of Occupants: ${occupants}
     //- Heating Type: ${heating_type}
     //- Cooling Type: ${cooling_type}
      
      //Provide practical and actionable suggestions to reduce energy consumption and save money.
     `;

      fetch("http://localhost:5000/api/huggingface", { //"https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt })
        })
          .then(res => res.json())
          .then(data => {
            // data might be an array or object; handle accordingly
            console.log("Response from local server:", data);
          })
          .catch(err => console.error("Error calling local server:", err));
        })

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
