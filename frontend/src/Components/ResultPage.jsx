import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

const ResultPage = () => {
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
    <div className="result-page">
      <h1>Your Monthly Energy Consumption: {prediction} KWh</h1>
      <h2>Suggestions</h2>
      <p>{suggestion}</p>
    </div>
  );
}

export default ResultPage;
