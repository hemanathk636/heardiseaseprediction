// Frontend Client Script connecting to the FastAPI Backend
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("prediction-form");
  const resultCard = document.getElementById("result-card");
  const riskBadge = document.getElementById("risk-badge");
  const riskProbText = document.getElementById("risk-probability");
  const progressFill = document.getElementById("progress-fill");
  const recText = document.getElementById("recommendation-text");
  const resetBtn = document.getElementById("reset-btn");
  const submitBtn = document.getElementById("submit-btn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect and format patient input values
    const formData = new FormData(form);
    const payload = {
      age: parseInt(formData.get("age")),
      sex: parseInt(formData.get("sex")),
      cp: parseInt(formData.get("cp")),
      trestbps: parseFloat(formData.get("trestbps")),
      chol: parseFloat(formData.get("chol")),
      fbs: parseInt(formData.get("fbs")),
      restecg: parseInt(formData.get("restecg")),
      thalach: parseFloat(formData.get("thalach")),
      exang: parseInt(formData.get("exang")),
      oldpeak: parseFloat(formData.get("oldpeak")),
      slope: parseInt(formData.get("slope")),
      ca: parseInt(formData.get("ca")),
      thal: parseInt(formData.get("thal"))
    };

    submitBtn.disabled = true;
    submitBtn.innerHTML = "<span>Analyzing Biomarkers...</span>";

    try {
      // Send HTTP POST request to the Backend API
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`);
      }

      const data = await response.json();
      displayResults(data);

    } catch (error) {
      console.error("Prediction request failed:", error);
      alert("Error communicating with backend API. Ensure the server is running on localhost:8000.");
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = "<span>Run AI Risk Evaluation</span>";
    }
  });

  function displayResults(data) {
    const isHighRisk = data.prediction === 1;

    // Show result container
    resultCard.classList.remove("hidden");
    resultCard.scrollIntoView({ behavior: "smooth" });

    // Update risk badge
    riskBadge.textContent = isHighRisk ? "⚠️ High Risk Detected" : "✅ Low Risk / Normal Range";
    riskBadge.className = `badge ${isHighRisk ? "danger" : "success"}`;

    // Update progress bar & probability
    riskProbText.textContent = `${data.probability}%`;
    progressFill.style.width = `${data.probability}%`;
    progressFill.style.backgroundColor = isHighRisk ? "#ef4444" : "#10b981";

    // Update clinical recommendation
    recText.textContent = data.recommendation;
  }

  resetBtn.addEventListener("click", () => {
    resultCard.classList.add("hidden");
    form.scrollIntoView({ behavior: "smooth" });
  });
});
