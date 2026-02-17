// ============================================================
// TomatoQualityAI — Frontend Script
// Connects to the Gradio API on Hugging Face Spaces
// ============================================================

const API_BASE = "https://swayamshetkar-tomato-quality-detector.hf.space/gradio_api";

// DOM Elements
const webcamVideo = document.getElementById("webcam");
const captureCanvas = document.getElementById("capture-canvas");
const webcamOverlay = document.getElementById("webcam-overlay");
const webcamContainer = document.getElementById("webcam-container");
const uploadContainer = document.getElementById("upload-container");
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const uploadPreview = document.getElementById("upload-preview");
const detectBtn = document.getElementById("detect-btn");
const btnText = document.querySelector(".btn-text");
const btnLoader = document.querySelector(".btn-loader");
const resultPlaceholder = document.getElementById("result-placeholder");
const resultImage = document.getElementById("result-image");
const resultCard = document.getElementById("result-card");
const resultLabel = document.getElementById("result-label");
const resultInfo = document.getElementById("result-info");
const fpsBadge = document.getElementById("fps-badge");
const autoDetectCheckbox = document.getElementById("auto-detect");

let webcamStream = null;
let autoDetectInterval = null;
let isProcessing = false;
let currentMode = "webcam";
let currentUploadedFile = null;

// ============================================================
// Mode Switching
// ============================================================
function switchMode(mode) {
    currentMode = mode;
    document.getElementById("btn-webcam").classList.toggle("active", mode === "webcam");
    document.getElementById("btn-upload").classList.toggle("active", mode === "upload");

    if (mode === "webcam") {
        webcamContainer.classList.remove("hidden");
        uploadContainer.classList.add("hidden");
        detectBtn.disabled = !webcamStream;
    } else {
        webcamContainer.classList.add("hidden");
        uploadContainer.classList.remove("hidden");
        detectBtn.disabled = !currentUploadedFile;
        // Stop auto-detect when switching to upload
        if (autoDetectCheckbox.checked) {
            autoDetectCheckbox.checked = false;
            toggleAutoDetect();
        }
    }
}

// ============================================================
// Webcam
// ============================================================
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } }
        });
        webcamVideo.srcObject = webcamStream;
        webcamOverlay.classList.add("hidden");
        detectBtn.disabled = false;
    } catch (err) {
        console.error("Webcam access denied:", err);
        alert("Could not access webcam. Please allow camera permissions.");
    }
}

function captureFrame() {
    const ctx = captureCanvas.getContext("2d");
    captureCanvas.width = webcamVideo.videoWidth;
    captureCanvas.height = webcamVideo.videoHeight;
    ctx.drawImage(webcamVideo, 0, 0);
    return captureCanvas.toDataURL("image/jpeg", 0.85);
}

// ============================================================
// Upload
// ============================================================
uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        alert("Please upload an image file.");
        return;
    }
    currentUploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadPreview.src = e.target.result;
        uploadPreview.classList.remove("hidden");
        detectBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ============================================================
// Auto-Detect Toggle
// ============================================================
function toggleAutoDetect() {
    if (autoDetectCheckbox.checked && currentMode === "webcam" && webcamStream) {
        autoDetectInterval = setInterval(() => {
            if (!isProcessing) runDetection();
        }, 1500); // Process every 1.5 seconds
    } else {
        clearInterval(autoDetectInterval);
        autoDetectInterval = null;
    }
}

// ============================================================
// Detection via Gradio API
// ============================================================

/**
 * Convert a base64 data URL to a Blob for proper file upload.
 */
function base64ToBlob(base64DataUrl) {
    const parts = base64DataUrl.split(",");
    const mime = parts[0].match(/:(.*?);/)[1];
    const byteString = atob(parts[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mime });
}

async function runDetection() {
    if (isProcessing) return;
    isProcessing = true;

    // UI: loading state
    btnText.classList.add("hidden");
    btnLoader.classList.remove("hidden");
    detectBtn.disabled = true;
    detectBtn.classList.add("detecting");

    const startTime = performance.now();

    try {
        let base64Image;

        if (currentMode === "webcam") {
            base64Image = captureFrame();
        } else if (currentUploadedFile) {
            base64Image = await fileToBase64(currentUploadedFile);
        } else {
            throw new Error("No image available");
        }

        // Step 1: Upload the image as a file to the Gradio API
        const blob = base64ToBlob(base64Image);
        const formData = new FormData();
        formData.append("files", blob, "image.jpg");

        const uploadRes = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });

        if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status}`);

        const uploadedFiles = await uploadRes.json();
        const imagePath = uploadedFiles[0];

        // Step 2: Call predict endpoint
        const callRes = await fetch(`${API_BASE}/call/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                data: [{ path: imagePath, meta: { _type: "gradio.FileData" } }]
            })
        });

        if (!callRes.ok) throw new Error(`API call failed: ${callRes.status}`);

        const callData = await callRes.json();
        const eventId = callData.event_id;

        // Step 3: Get results via SSE event stream
        const result = await getEventResult(eventId);

        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        const fps = (1000 / (performance.now() - startTime)).toFixed(1);

        // Parse result
        if (result && result.length >= 2) {
            const outputImage = result[0];
            const conclusion = result[1];

            // Show detection image
            if (outputImage && outputImage.url) {
                resultImage.src = outputImage.url;
            } else if (outputImage && outputImage.path) {
                resultImage.src = `${API_BASE}/file=${outputImage.path}`;
            } else if (typeof outputImage === "string") {
                resultImage.src = outputImage;
            }
            resultImage.classList.remove("hidden");
            resultPlaceholder.classList.add("hidden");

            // Show conclusion
            resultLabel.textContent = conclusion;
            resultLabel.className = "result-label";
            if (conclusion.toLowerCase().includes("ripe") && !conclusion.toLowerCase().includes("unripe")) {
                resultLabel.classList.add("ripe");
            } else if (conclusion.toLowerCase().includes("unripe")) {
                resultLabel.classList.add("unripe");
            } else if (conclusion.toLowerCase().includes("damaged")) {
                resultLabel.classList.add("damaged");
            } else {
                resultLabel.classList.add("none");
            }
            resultInfo.textContent = `Processed in ${elapsed}s`;
            resultCard.classList.remove("hidden");

            // Show FPS
            fpsBadge.textContent = `${fps} FPS`;
            fpsBadge.classList.remove("hidden");
        }

    } catch (err) {
        console.error("Detection error:", err);
        resultLabel.textContent = "Error";
        resultLabel.className = "result-label none";
        resultInfo.textContent = err.message;
        resultCard.classList.remove("hidden");
    } finally {
        isProcessing = false;
        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
        detectBtn.disabled = false;
        detectBtn.classList.remove("detecting");
    }
}

// ============================================================
// Fetch Event Result from Gradio SSE
// ============================================================
async function getEventResult(eventId) {
    return new Promise((resolve, reject) => {
        const url = `${API_BASE}/call/predict/${eventId}`;
        let resolved = false;

        fetch(url).then(response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            function read() {
                reader.read().then(({ done, value }) => {
                    if (resolved) return;

                    buffer += decoder.decode(value, { stream: true });

                    // Process complete SSE messages (separated by double newlines)
                    const messages = buffer.split("\n");
                    let currentEvent = "";

                    for (let i = 0; i < messages.length - 1; i++) {
                        const line = messages[i].trim();

                        if (line.startsWith("event:")) {
                            currentEvent = line.substring(6).trim();
                        } else if (line.startsWith("data:")) {
                            const dataStr = line.substring(5).trim();

                            if (currentEvent === "error") {
                                resolved = true;
                                reject(new Error(`Gradio error: ${dataStr}`));
                                return;
                            }

                            // "complete" event has the final result as a JSON array
                            if (currentEvent === "complete") {
                                try {
                                    const data = JSON.parse(dataStr);
                                    resolved = true;
                                    resolve(data);
                                    return;
                                } catch (e) {
                                    // Not valid JSON, keep reading
                                }
                            }

                            // Also try parsing any data line for a direct array result
                            try {
                                const data = JSON.parse(dataStr);
                                if (Array.isArray(data) && data.length >= 2) {
                                    resolved = true;
                                    resolve(data);
                                    return;
                                }
                            } catch (e) {
                                // Not JSON yet, keep reading
                            }
                        }
                    }

                    // Keep last partial line in buffer
                    buffer = messages[messages.length - 1];

                    if (done) {
                        if (!resolved) {
                            reject(new Error("Stream ended without result"));
                        }
                        return;
                    }

                    read();
                }).catch(err => {
                    if (!resolved) reject(err);
                });
            }
            read();
        }).catch(err => {
            if (!resolved) reject(err);
        });

        // Timeout after 30 seconds
        setTimeout(() => {
            if (!resolved) {
                resolved = true;
                reject(new Error("Detection timed out"));
            }
        }, 30000);
    });
}

// ============================================================
// Utility
// ============================================================
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}
