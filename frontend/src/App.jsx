import React from "react";
import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

// Import page components
import WelcomePage from "./Components/WelcomePage";
import DataInput from "./Components/DataInput";
import ResultPage from "./Components/ResultPage";

const slideVariants = {
    initial: {y: "100%", opacity: 0},
    animate: {y: "0%", opacity: 1},
    exit: {y: "-100%", opacity: 0},
};

function AnimatedRoutes(){
    const location = useLocation();

    return (
        <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
            <Route
            path="/"
            element={
                <motion.div
                variants={slideVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                transition={{ duration: 0.5 }}
                >
                <WelcomePage />
                </motion.div>
            }
            />
            <Route
            path="/data-input"
            element={
                <motion.div
                variants={slideVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                transition={{ duration: 0.5 }}
                >
                <DataInput />
                </motion.div>
            }
            />
            <Route
            path="/results"
            element={
                <motion.div
                variants={slideVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                transition={{ duration: 0.5 }}
                >
                <ResultPage />
                </motion.div>
            }
            />
        </Routes>
        </AnimatePresence>
    );
    }


    function App() {
        return (
          <Router>
            <AnimatedRoutes />
          </Router>
        );
      }
      
      export default App;