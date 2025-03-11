import React, { useState } from "react";
import StepWizard from "react-step-wizard";
import { useNavigate } from "react-router-dom";
import "../Styles/DataInputWizard.css";

// Import images for each step's background
import memoryImage from "../Styles/Images/the-persistence-of-memory-1931.jpg";
import venereImage from "../Styles/Images/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg";
import jacquesImage from "../Styles/Images/17Louis-David-Review-tennis-superJumbo.jpg";
import danteBoat from "../Styles/Images/Eugène_Delacroix_-_The_Barque_of_Dante.jpg";
import starryNight from "../Styles/Images/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg";

/** 
 * Background array for each step, 
 * index: 0 => Step1, 1 => Step2, 2 => Step3, 3 => Step4, 4 => Step5
 */
const backgrounds = [
  memoryImage,  
  venereImage,  
  jacquesImage, 
  danteBoat,    
  starryNight,  
];

/* 
  STEP 1: Enter the Month 
  - Local error handling: if user hasn't selected a month, display an error and do NOT call nextStep().
*/
function Step1({ nextStep, formData, setFormData }) {
  const [error, setError] = useState("");

  const handleNext = () => {
    if (!formData.month || formData.month.trim() === "") {
      setError("Please select a month.");
      return;
    }
    setError("");
    nextStep();
  };

  return (
    <div className="wizard-step">
      <h2>Enter the Month</h2>
      <select
        name="month"
        value={formData.month}
        onChange={(e) => setFormData({ ...formData, month: e.target.value })}
      >
        <option value="">Select</option>
        {[
          "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"
        ].map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      {error && <p className="error-message">{error}</p>}
      <button onClick={handleNext} className="brown-button">Next</button>
    </div>
  );
}

/* 
  STEP 2: Enter House Size 
  - Validates that "size" is not empty.
*/
function Step2({ previousStep, nextStep, formData, setFormData }) {
  const [error, setError] = useState("");

  const handleNext = () => {
    if (!formData.size || formData.size.trim() === "") {
      setError("Please enter the house size.");
      return;
    }
    setError("");
    nextStep();
  };

  return (
    <div className="wizard-step">
      <h2>Enter House Size (sq ft)</h2>
      <input
        type="number"
        name="size"
        value={formData.size}
        onChange={(e) => setFormData({ ...formData, size: e.target.value })}
        placeholder="House Size"
      />
      {error && <p className="error-message">{error}</p>}
      <div className="button-group">
      <button onClick={previousStep} className="creme-button">Back</button>
        <button onClick={handleNext} className="creme-button">Next</button>
      </div>
    </div>
  );
}

/* 
  STEP 3: Enter Number of Occupants
*/
function Step3({ previousStep, nextStep, formData, setFormData }) {
  const [error, setError] = useState("");

  const handleNext = () => {
    if (!formData.occupants || formData.occupants.trim() === "") {
      setError("Please enter the number of occupants.");
      return;
    }
    setError("");
    nextStep();
  };

  return (
    <div className="wizard-step">
      <h2>Enter Number of Occupants</h2>
      <input
        type="number"
        name="occupants"
        value={formData.occupants}
        onChange={(e) => setFormData({ ...formData, occupants: e.target.value })}
        placeholder="Occupants"
      />
      {error && <p className="error-message">{error}</p>}
      <div className="button-group">
        <button onClick={previousStep} className="yellow-button">Back</button>
        <button onClick={handleNext} className="yellow-button">Next</button>
      </div>
    </div>
  );
}

/* 
  STEP 4: Select Heating Type
*/
function Step4({ previousStep, nextStep, formData, setFormData }) {
  const [error, setError] = useState("");

  const handleNext = () => {
    if (!formData.heating_type || formData.heating_type.trim() === "") {
      setError("Please select a heating type.");
      return;
    }
    setError("");
    nextStep();
  };

  return (
    <div className="wizard-step">
      <h2>Select Heating Type</h2>
      <select
        name="heating_type"
        value={formData.heating_type}
        onChange={(e) => setFormData({ ...formData, heating_type: e.target.value })}
      >
        <option value="">Select</option>
        {["Electric", "Gas", "Solar", "None"].map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      {error && <p className="error-message">{error}</p>}
      <div className="button-group">
        <button onClick={previousStep} className="red-button">Back</button>
        <button onClick={handleNext} className="red-button">Next</button>
      </div>
    </div>
  );
}

/* 
  STEP 5: Select Cooling Type & Submit
*/
function Step5({ previousStep, formData, setFormData }) {
  const navigate = useNavigate();
  const [error, setError] = useState("");

  const finishWizard = () => {
    if (!formData.cooling_type || formData.cooling_type.trim() === "") {
      setError("Please select a cooling type.");
      return;
    }
    setError("");

    // Final step: call your API, then navigate to results
    fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        navigate("/results", { state: { prediction: data["KWh Consumption"], ...formData } });
      })
      .catch((error) => {
        console.error("Error fetching prediction:", error);
      });
  };

  return (
    <div className="wizard-step">
      <h2>Select Cooling Type</h2>
      <select
        name="cooling_type"
        value={formData.cooling_type}
        onChange={(e) => setFormData({ ...formData, cooling_type: e.target.value })}
      >
        <option value="">Select</option>
        {["Central AC", "Fans", "None"].map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      {error && <p className="error-message">{error}</p>}
      <div className="button-group">
        <button onClick={previousStep} className="blue-button">Back</button>
        <button onClick={finishWizard} className="blue-button">Get Predictions!</button>
      </div>
    </div>
  );
}

function DataInputWizard() {
  // For user input
  const [formData, setFormData] = useState({
    month: "",
    size: "",
    occupants: "",
    heating_type: "",
    cooling_type: "",
  });

  // For tracking the current step in order to change backgrounds
  const [activeStep, setActiveStep] = useState(1);

  // Called by react-step-wizard whenever the step changes
  const onStepChange = (stats) => {
    setActiveStep(stats.activeStep);
  };

  // Choose the correct background for the current step
  const currentBackground = backgrounds[activeStep - 1] || backgrounds[0];

  return (
    <div
      className="data-input-wizard-container"
      style={{
        backgroundImage: `linear-gradient(rgba(255,255,255,0.2), rgba(255,255,255,0.7)), url(${currentBackground})`,
        backgroundSize: "cover",
        backgroundPosition: "center 15%",
        backgroundRepeat: "no-repeat",
      }}
    >
      <StepWizard onStepChange={onStepChange} transitions={{ enterRight: "slideInRight", exitLeft: "slideOutLeft" }}>
        <Step1 formData={formData} setFormData={setFormData} />
        <Step2 formData={formData} setFormData={setFormData} />
        <Step3 formData={formData} setFormData={setFormData} />
        <Step4 formData={formData} setFormData={setFormData} />
        <Step5 formData={formData} setFormData={setFormData} />
      </StepWizard>
    </div>
  );
}

export default DataInputWizard;
