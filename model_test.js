const imageInput = document.getElementById("image-input");
const predictButton = document.getElementById("predict-btn");
const previewImage = document.getElementById("preview-image");
const statusText = document.getElementById("status-text");
const topEmotion = document.getElementById("top-emotion");
const topConfidence = document.getElementById("top-confidence");
const cropUsed = document.getElementById("crop-used");
const probabilityTableBody = document.querySelector("#probability-table tbody");
const modelInput = document.getElementById("model-input");
const labelsInput = document.getElementById("labels-input");
const uploadModelButton = document.getElementById("upload-model-btn");
const modelUploadStatus = document.getElementById("model-upload-status");
const modelLoaded = document.getElementById("model-loaded");
const modelOutputCount = document.getElementById("model-output-count");
const modelLabelCount = document.getElementById("model-label-count");
const modelPath = document.getElementById("model-path");

function setStatus(message) {
    statusText.textContent = message;
}

function setModelUploadStatus(message) {
    modelUploadStatus.textContent = message;
}

function clearTable() {
    probabilityTableBody.innerHTML = "";
}

function renderProbabilities(probabilities) {
    clearTable();

    Object.entries(probabilities).forEach(([emotion, value]) => {
        const row = document.createElement("tr");
        const emotionCell = document.createElement("td");
        const valueCell = document.createElement("td");

        emotionCell.textContent = emotion;
        valueCell.textContent = `${(Number(value) * 100).toFixed(2)}%`;

        row.appendChild(emotionCell);
        row.appendChild(valueCell);
        probabilityTableBody.appendChild(row);
    });
}

function showPreview(file) {
    const imageUrl = URL.createObjectURL(file);
    previewImage.src = imageUrl;
    previewImage.style.display = "block";
}

async function predictImage() {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
        setStatus("Pilih gambar dulu.");
        return;
    }

    setStatus("Memproses...");
    predictButton.disabled = true;

    try {
        const formData = new FormData();
        formData.append("image", file);

        const response = await fetch("/predict_image", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Prediksi gagal.");
        }

        topEmotion.textContent = data.emotion;
        topConfidence.textContent = `${(Number(data.confidence) * 100).toFixed(2)}%`;
        cropUsed.textContent = data.used_face_crop ? "yes" : "no";
        renderProbabilities(data.probabilities || {});
        setStatus(data.used_face_crop ? "Prediksi selesai dari crop wajah." : "Prediksi selesai dari gambar utuh.");
    } catch (error) {
        setStatus(error.message || "Terjadi error.");
    } finally {
        predictButton.disabled = false;
    }
}

function renderModelStatus(data) {
    modelLoaded.textContent = data.model_loaded ? "yes" : "no";
    modelOutputCount.textContent = data.output_count ?? "-";
    modelLabelCount.textContent = data.class_count ?? "-";
    modelPath.textContent = data.model_path || "-";
}

async function refreshModelStatus() {
    try {
        const response = await fetch("/model_status", {
            cache: "no-store"
        });
        const data = await response.json();
        renderModelStatus(data);
        setModelUploadStatus(data.model_loaded ? "Model siap." : "Model belum loaded.");
    } catch (error) {
        setModelUploadStatus("Gagal membaca status model.");
    }
}

async function uploadModel() {
    const modelFile = modelInput.files && modelInput.files[0];
    const labelsFile = labelsInput.files && labelsInput.files[0];

    if (!modelFile) {
        setModelUploadStatus("Pilih file model .h5/.keras dulu.");
        return;
    }

    setModelUploadStatus("Uploading dan reload model...");
    uploadModelButton.disabled = true;

    try {
        const formData = new FormData();
        formData.append("model", modelFile);

        if (labelsFile) {
            formData.append("labels", labelsFile);
        }

        const response = await fetch("/upload_model", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Upload model gagal.");
        }

        renderModelStatus(data);
        setModelUploadStatus(data.message || "Model aktif.");
    } catch (error) {
        setModelUploadStatus(error.message || "Upload model gagal.");
    } finally {
        uploadModelButton.disabled = false;
    }
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
        return;
    }
    showPreview(file);
});

predictButton.addEventListener("click", predictImage);
uploadModelButton.addEventListener("click", uploadModel);
refreshModelStatus();
