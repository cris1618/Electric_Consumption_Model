import React from "react";
import { useLocation } from "react-router-dom";

const ResultPage = () => {
  const { state } = useLocation();
  const { prediction } = state || {};

  return (
    <div className="result-container">
      <h1>Your Monthly Energy Consumption</h1>
      {prediction ? (
        <>
          <p>{prediction} KWh</p>
          <p>
            {/* You can add some suggestions based on the prediction here */}
            Consider ways to improve energy efficiency in your home!
          </p>
        </>
      ) : (
        <p>No prediction available.</p>
      )}
    </div>
  );
};

export default ResultPage;
