import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import ImagePreview from "./components/ImagePreview";
import ProcessingSteps from "./components/ProcessingSteps";
import "./App.css";

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);

  const handleImageUpload = (file) => {
    setUploadedImage(file);
    setCurrentStep(0);
  };

  const handleStartProcessing = () => {
    setCurrentStep(1);
  };

  return (
    <div className="app">

      <header className="header">
        <h1>Infrared Image Enhancement & Colorization</h1>

        <p>
          Transform low-visibility infrared satellite images
          into enhanced and realistic RGB images.
        </p>
      </header>

      <main className="main-container">

        <section className="upload-section">
          <h2>Upload Infrared Satellite Image</h2>

          <p>
            Upload an infrared satellite image to begin processing.
          </p>

          <ImageUploader
            onImageUpload={handleImageUpload}
          />

        </section>

        {uploadedImage && (
          <>
            <ImagePreview image={uploadedImage} />

            <section className="process-section">
              <button
                className="process-button"
                onClick={handleStartProcessing}
              >
                Start Processing
              </button>
            </section>

            {currentStep > 0 && (
              <ProcessingSteps
                currentStep={currentStep}
              />
            )}
          </>
        )}

      </main>
    </div>
  );
}

export default App;