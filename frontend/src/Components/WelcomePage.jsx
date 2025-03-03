import React from "react";
import { useNavigate } from "react-router-dom";
import "../Styles/WelcomePage.css";

const WelcomePage = () => {
    const navigate = useNavigate();
  
    const handleStart = () => {
      navigate("/data-input");
    };
  
    return (
      <div className="welcome-container">
        {/* 
          A translucent box that holds the text 
          to ensure readability over the background image 
        */}
        <div className="welcome-text-box">
          <h1>Welcome to the Energy Consumption Predictor!</h1>
          <p>
            By
            inputting a few simple details, you'll receive a personalized estimate
            of your monthly energy usage and suggestions on how to reduce it and
            save money. Click the button below to get started!
          </p>
          <button className="start-button" onClick={handleStart}>
            Get Started
          </button>
        </div>
      </div>
    );
  };
  
  export default WelcomePage;