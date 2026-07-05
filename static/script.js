const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewArea = document.getElementById("previewArea");
const previewImg = document.getElementById("previewImg");
const changeBtn = document.getElementById("changeBtn");
const spinner = document.getElementById("spinner");
const results = document.getElementById("results");
const resultCard = document.getElementById("resultCard");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const realScore = document.getElementById("realScore");
const fakeScore = document.getElementById("fakeScore");

const ALLOWED = [".png", ".jpg", ".jpeg", ".webp"];

function isValidFile(name) {
  const ext = name?.substring(name.lastIndexOf(".")).toLowerCase();
  return ALLOWED.includes(ext);
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("border-indigo-400", "bg-white/10");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("border-indigo-400", "bg-white/10");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("border-indigo-400", "bg-white/10");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

changeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  resetPreview();
  fileInput.click();
});

function resetPreview() {
  uploadPrompt.classList.remove("hidden");
  previewArea.classList.add("hidden");
  results.classList.add("hidden");
  resultCard.className = "rounded-2xl p-6 text-center transition-all duration-500 bg-white/5 border border-white/10";
  previewImg.src = "";
}

function handleFile(file) {
  if (!isValidFile(file.name)) {
    alert("Only PNG, JPG, JPEG, and WEBP files are allowed.");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadPrompt.classList.add("hidden");
    previewArea.classList.remove("hidden");
    results.classList.add("hidden");
  };
  reader.readAsDataURL(file);

  classifyImage(file);
}

async function classifyImage(file) {
  spinner.classList.remove("hidden");
  results.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || "Prediction failed.");
      spinner.classList.add("hidden");
      return;
    }
    const data = await res.json();

    const top = data.top_label;
    const conf = data.top_confidence;
    const preds = data.predictions || [];

    const real = preds.find((p) => p.label.toLowerCase() === "real");
    const fake = preds.find((p) => p.label.toLowerCase().includes("fake") || p.label.toLowerCase().includes("ai"));

    realScore.textContent = (real ? real.confidence : 0) + "%";
    fakeScore.textContent = (fake ? fake.confidence : 0) + "%";

    resultLabel.textContent = top;
    resultConfidence.textContent = conf + "% confidence";

    resultCard.className =
      "rounded-2xl p-6 text-center transition-all duration-500 border " +
      (top.toLowerCase() === "real"
        ? "bg-green-900/30 border-green-500/50 text-green-300"
        : "bg-red-900/30 border-red-500/50 text-red-300");

    results.classList.remove("hidden");
  } catch (err) {
    alert("Network error. Please try again.");
  } finally {
    spinner.classList.add("hidden");
  }
}
