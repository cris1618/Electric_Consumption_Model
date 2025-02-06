import React, { useState } from "react";

function EnergyPredictor() {
  const [season, setSeason] = useState("Winter");
  const [size, setSize] = useState("");
  const [occupants, setOccupants] = useState("");
  const [heatingType, setHeatingType] = useState("Electric");
  const [coolingType, setCoolingType] = useState("None");
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    const requestData = {
      season,
      size: parseInt(size),
      occupants: parseInt(occupants),
      heating_type: heatingType,
      cooling_type: coolingType,
    };
  
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });
  
      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }
  
      const data = await response.json();
      console.log("API Response:", data);  
  
      setPrediction(data["KWh Consumption"]);  
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };
  return (
    <div>
      <h1>Energy Consumption Predictor</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Season:
          <select value={season} onChange={(e) => setSeason(e.target.value)}>
            <option>Winter</option>
            <option>Spring</option>
            <option>Summer</option>
            <option>Fall</option>
          </select>
        </label>

        <label>
          House Size (sq ft):
          <input
            type="number"
            value={size}
            onChange={(e) => setSize(e.target.value)}
          />
        </label>

        <label>
          Number of Occupants:
          <input
            type="number"
            value={occupants}
            onChange={(e) => setOccupants(e.target.value)}
          />
        </label>

        <label>
          Heating Type:
          <select value={heatingType} onChange={(e) => setHeatingType(e.target.value)}>
            <option>Electric</option>
            <option>Gas</option>
            <option>Solar</option>
            <option>None</option>
          </select>
        </label>

        <label>
          Cooling Type:
          <select value={coolingType} onChange={(e) => setCoolingType(e.target.value)}>
            <option>Central AC</option>
            <option>Fans</option>
            <option>None</option>
          </select>
        </label>

        <button type="submit">Predict</button>
      </form>

      {prediction !== null && <h2>Estimated Consumption: {prediction} KWh</h2>}
    </div>
  );
}

export default EnergyPredictor;
