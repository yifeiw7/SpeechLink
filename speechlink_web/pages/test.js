const startStopButton = document.getElementById("startStopButton");
const uploadButton = document.getElementById("uploadButton");
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");
const audioPlayer = document.getElementById("audioPlayer");
const loadingIndicator = document.getElementById("loadingIndicator");

let mediaRecorder;
let audioChunks = [];
let listening = false;
let uploadMode = false;

// Toggle between recording and upload modes
startStopButton.addEventListener("click", () => {
    if (uploadMode) {
        // Handle file upload if in upload mode
        fileInput.click();
    } else {
        // Handle recording
        if (listening) {
            mediaRecorder.stop();
            listening = false;
            startStopButton.textContent = "🎤 Start recording";
        } else {
            startRecording();
            listening = true;
            startStopButton.textContent = "⏹ Stop recording";
        }
    }
});

// Toggle upload mode on click
uploadButton.addEventListener("click", () => {
    uploadMode = !uploadMode;
    startStopButton.classList.toggle("upload-mode", uploadMode);
    startStopButton.textContent = uploadMode ? "Select File" : "Record Audio";
});

// Start recording function
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            audioChunks = [];
            sendAudioToServer(audioBlob);
        };
    });
}

// Handle file selection and upload
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file && (file.type === "audio/wav" || file.type === "audio/mp3")) {
        sendAudioToServer(file);
        resultDiv.innerHTML = "Transcribing uploaded audio...";
    } else {
        resultDiv.innerHTML = "Please upload a valid .wav or .mp3 file.";
    }
});

// Function to send audio file to the server
function sendAudioToServer(audioBlob) {
    loadingIndicator.style.display = "block"; // Show loading indicator
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");

    fetch("http://localhost:5001/transcribe", {
        method: "POST",
        body: formData,
    })
    .then((response) => {
        loadingIndicator.style.display = "none"; // Hide loading indicator
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then((data) => {
        if (data.error) {
            throw new Error(data.error);
        }
        resultDiv.innerHTML = `Transcription: ${data.transcript}`;
        audioPlayer.src = URL.createObjectURL(audioBlob);
        audioPlayer.style.display = "block";
    })
    .catch((error) => {
        console.error("Error:", error);
        resultDiv.innerHTML = `An error occurred during transcription: ${error.message}`;
        loadingIndicator.style.display = "none";
    });
}