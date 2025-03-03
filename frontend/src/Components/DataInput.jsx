import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import "../Styles/DataInput.css"; 
import memoryImage from "../Styles/Images/the-persistence-of-memory-1931.jpg";
import venereImage from "../Styles/Images/Sandro_Botticelli_-_La_nascita_di_Venere_-_Google_Art_Project_-_edited.jpg";
import JacuquesImage from "../Styles/Images/17Louis-David-Review-tennis-superJumbo.jpg"

// Define the steps of your multi-step form
const stepsData = [
  {
    label: "Enter the Month",
    type: "select",
    name: "month",
    options: [
      "January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"
    ],
    backgroundImage: memoryImage,
    buttonClass: "brown-button"
  },
  {
    label: "Enter House Size (sq ft)",
    type: "number",
    name: "size",
    backgroundImage: venereImage,
    buttonClass: "creme-button"
  },
  {
    label: "Enter Number of Occupants",
    type: "number",
    name: "occupants",
    backgroundImage: JacuquesImage,
    buttonClass: "yellow-button"
  },
  {
    label: "Select Heating Type",
    type: "select",
    name: "heating_type",
    options: ["Electric", "Gas", "Solar", "None"],
  },
  {
    label: "Select Cooling Type",
    type: "select",
    name: "cooling_type",
    options: ["Central AC", "Fans", "None"],
  },
];

function DataInput() {
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState({
    month: "",
    size: "",
    occupants: "",
    heating_type: "",
    cooling_type: "",
  });
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleNext = () => {
    const currentField = stepsData[currentStep].name;
    if (!formData[currentField] || formData[currentField].toString().trim() === "") {
      setError("Please enter a value.");
      return;
    } else {
      setError("");
    }
    // If it's not the last step, advance to the next one.
    if (currentStep < stepsData.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      })
        .then((response) => response.json())
        .then((data) => {
          // Navigate to results page, passing the prediction via state
          navigate("/results", { state: { prediction: data["KWh Consumption"], ...formData } });
        })
        .catch((error) => {
          console.error("Error fetching prediction:", error);
        });
      }
    }
  
    
  // Get the data for the current step
  const currentStepData = stepsData[currentStep];

  console.log("Current Step:", currentStep);
  console.log("Background Image:", currentStepData.backgroundImage);
  return (
    <div className="data-input-container"
    style={{
            backgroundImage: `linear-gradient(rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.7)), url(${currentStepData.backgroundImage})`,
            backgroundSize: "cover",
            backgroundPosition: "center 15%", 
            backgroundRepeat: "no-repeat",
        }}
      >
      <AnimatePresence exitBeforeEnter>
        <motion.div
          key={currentStep}
          className="input-step"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -50 }}
          transition={{ duration: 0.3 }}
        >
          <h2>{currentStepData.label}</h2>
          {currentStepData.type === "select" ? (
            <select
              name={currentStepData.name}
              value={formData[currentStepData.name]}
              onChange={handleChange}
            >
              <option value="">Select</option>
              {currentStepData.options.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          ) : (
            <input
              type={currentStepData.type}
              name={currentStepData.name}
              value={formData[currentStepData.name]}
              onChange={handleChange}
              placeholder={currentStepData.label}
            />
          )}
          {error && <p className="error-message">{error}</p>}
          <button className={currentStepData.buttonClass} onClick={handleNext}>
              Next
          </button>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export default DataInput;
