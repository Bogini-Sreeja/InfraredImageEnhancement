function ProcessingSteps({ currentStep }) {
  const steps = [
    "Image Preprocessing",
    "Image Enhancement",
    "IR → RGB Colorization",
    "Evaluation",
  ];

  return (
    <div className="processing-container">
      <h2>Processing Pipeline</h2>

      <div className="steps">
        {steps.map((step, index) => {
          const stepNumber = index + 1;

          let status = "pending";

          if (stepNumber < currentStep) {
            status = "completed";
          } else if (stepNumber === currentStep) {
            status = "active";
          }

          return (
            <div
              key={step}
              className={`step ${status}`}
            >
              <div className="step-number">
                {status === "completed" ? "✓" : stepNumber}
              </div>

              <span>{step}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default ProcessingSteps;