import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);

  const [originalPreview, setOriginalPreview] = useState(null);
  const [preprocessedImage, setPreprocessedImage] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [superResolutionImage, setSuperResolutionImage] = useState(null);
  const [colorizedImage, setColorizedImage] = useState(null);

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");

  // =========================================
  // SELECT IMAGE
  // =========================================

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) {
      return;
    }

    setFile(selectedFile);

    // Show original image
    const previewURL = URL.createObjectURL(selectedFile);

    setOriginalPreview(previewURL);

    // Clear previous results
    setPreprocessedImage(null);
    setEnhancedImage(null);
    setSuperResolutionImage(null);
    setColorizedImage(null);

    setStatus("Image selected");
  };

  // =========================================
  // RUN COMPLETE PIPELINE
  // =========================================
  const processImage = async () => {
  if (!file) {
    alert("Please select an IR image first.");
    return;
  }

  setLoading(true);
  setStatus("Starting image processing...");

  try {

    // =====================================
    // STEP 1 — PREPROCESSING
    // =====================================

    setStatus("Running image preprocessing...");

    const formData = new FormData();

    formData.append("file", file);

    const preprocessResponse = await fetch(
      `${API_URL}/preprocess`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!preprocessResponse.ok) {
      throw new Error(
        `Preprocessing failed: ${preprocessResponse.status}`
      );
    }

    const preprocessData =
      await preprocessResponse.json();

    console.log(
      "Preprocessing response:",
      preprocessData
    );

    if (preprocessData.error) {
      throw new Error(preprocessData.error);
    }

    const processedFilename =
      preprocessData.filename;

    setPreprocessedImage(
      `${API_URL}${preprocessData.url}`
    );


    // =====================================
    // STEP 2 — IMAGE ENHANCEMENT
    // =====================================

    setStatus("Running image enhancement...");

    const enhanceResponse = await fetch(
      `${API_URL}/enhance?filename=${encodeURIComponent(
        processedFilename
      )}`,
      {
        method: "POST",
      }
    );

    if (!enhanceResponse.ok) {
      throw new Error(
        `Enhancement failed: ${enhanceResponse.status}`
      );
    }

    const enhanceData =
      await enhanceResponse.json();

    console.log(
      "Enhancement response:",
      enhanceData
    );

    if (enhanceData.error) {
      throw new Error(enhanceData.error);
    }

    const enhancedFilename =
      enhanceData.filename;

    setEnhancedImage(
      `${API_URL}${enhanceData.url}`
    );


    // =====================================
    // STEP 3 — SUPER RESOLUTION
    // =====================================

    setStatus("Running super-resolution...");

    const srResponse = await fetch(
      `${API_URL}/super-resolution?filename=${encodeURIComponent(
        enhancedFilename
      )}`,
      {
        method: "POST",
      }
    );

    if (!srResponse.ok) {
      throw new Error(
        `Super-resolution failed: ${srResponse.status}`
      );
    }

    // IMPORTANT:
    // srData must be created BEFORE using srData.filename

    const srData =
      await srResponse.json();

    console.log(
      "Super-resolution response:",
      srData
    );

    if (srData.error) {
      throw new Error(srData.error);
    }

    setSuperResolutionImage(
      `${API_URL}${srData.url}`
    );


    // =====================================
    // STEP 4 — RGB COLORIZATION
    // =====================================

    setStatus("Running RGB colorization...");

    const colorizeResponse = await fetch(
      `${API_URL}/colorize?filename=${encodeURIComponent(
        srData.filename
      )}`,
      {
        method: "POST",
      }
    );

    if (!colorizeResponse.ok) {
      throw new Error(
        `RGB colorization failed: ${colorizeResponse.status}`
      );
    }

    const colorizeData =
      await colorizeResponse.json();

    console.log(
      "Colorization response:",
      colorizeData
    );

    if (colorizeData.error) {
      throw new Error(colorizeData.error);
    }

    setColorizedImage(
      `${API_URL}${colorizeData.url}`
    );

    setStatus(
      "RGB colorization completed successfully!"
    );

  } catch (error) {

    console.error(
      "Image processing error:",
      error
    );

    setStatus(
      `Error: ${error.message}`
    );

  } finally {

    // Always enable the button again
    setLoading(false);

  }
};

      
      

      
       

    





  // =========================================
  // UI
  // =========================================

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">
        <h1>
          Infrared Image Enhancement
        </h1>

        <p>
          Preprocessing • Enhancement •
          Super-Resolution
        </p>
      </header>


      {/* UPLOAD SECTION */}

      <section className="upload-section">

        <h2>
          Upload Infrared Image
        </h2>

        <p className="description">
          Upload an infrared satellite image
          to process it through the enhancement
          pipeline.
        </p>

        <label className="upload-box">

          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />

          <span>
            {file
              ? file.name
              : "Choose an IR image"}
          </span>

        </label>

        <button
          className="process-button"
          onClick={processImage}
          disabled={!file || loading}
        >
          {loading
            ? "Processing..."
            : "Process Image"}
        </button>

        {status && (
          <p className="status">
            {status}
          </p>
        )}

      </section>


      {/* RESULTS */}

      <section className="results-section">

        <h2>
          Processing Results
        </h2>

        <div className="results-grid">

          {/* ORIGINAL */}

          {originalPreview && (
            <div className="result-card">

              <h3>
                Original IR Image
              </h3>

              <img
                src={originalPreview}
                alt="Original infrared"
              />

            </div>
          )}


          {/* PREPROCESSED */}

          {preprocessedImage && (
            <div className="result-card">

              <h3>
                Preprocessed Image
              </h3>

              <img
                src={preprocessedImage}
                alt="Preprocessed infrared"
              />

            </div>
          )}


          {/* ENHANCED */}

          {enhancedImage && (
            <div className="result-card">

              <h3>
                Enhanced Image
              </h3>

              <img
                src={enhancedImage}
                alt="Enhanced infrared"
              />

            </div>
          )}


          {/* SUPER RESOLUTION */}

          {superResolutionImage && (
            <div className="result-card">

              <h3>
                Super-Resolved Image
              </h3>

              <img
                src={superResolutionImage}
                alt="Super resolved infrared"
              />

            </div>
          )}
          {/* RGB COLORIZATION */}

{colorizedImage && (
  <div className="result-card">

    <h3>
      RGB Colorization
    </h3>

    <img
      src={colorizedImage}
      alt="RGB colorized image"
    />

  </div>
)}

        </div>

      </section>

    </div>
  );
}

export default App;