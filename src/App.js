import './App.css';
import React, { useRef, useState } from 'react';
import axios from 'axios';

const AudioTranscription = () => {
  const [listening, setListening] = useState(false);
  const [uploadMode, setUploadMode] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [loading, setLoading] = useState(false);
  
  const fileInputRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleStartStop = () => {
    if (uploadMode) {
      fileInputRef.current.click();
    } else {
      if (listening) {
        mediaRecorderRef.current.stop();
        setListening(false);
      } else {
        startRecording();
        setListening(true);
      }
    }
  };

  const handleToggleUploadMode = () => {
    setUploadMode(!uploadMode);
  };

  const startRecording = () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
          audioChunksRef.current = [];
          sendAudioToServer(audioBlob);
        };

        mediaRecorder.start();
      });
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === "audio/wav" || file.type === "audio/mp3")) {
      sendAudioToServer(file);
      setTranscription("Transcribing uploaded audio...");
    } else {
      setTranscription("Please upload a valid .wav or .mp3 file.");
    }
  };

  const sendAudioToServer = (audioBlob) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");

    axios.post("http://localhost:5001/transcribe", formData)
      .then((response) => {
        setLoading(false);
        if (!response.data) {
          throw new Error("No data in response");
        }
        setTranscription(`Transcription: ${response.data.transcript}`);
        audioPlayerRef.current.src = URL.createObjectURL(audioBlob);
      })
      .catch((error) => {
        console.error("Error:", error);
        setTranscription(`An error occurred during transcription: ${error.message}`);
        setLoading(false);
      });
  };

  return (
    <div className="audio-transcription">
      <button id="startStopButton" onClick={handleStartStop}>
        {uploadMode ? "Select File" : listening ? "⏹ Stop recording" : "🎤 Start recording"}
      </button>
      <button id="uploadButton" onClick={handleToggleUploadMode}>
        {uploadMode ? "Switch to Record Mode" : "Switch to Upload Mode"}
      </button>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
      {loading && <div id="loadingIndicator">Transcribing...</div>}
      <div id="result">{transcription}</div>
      <audio ref={audioPlayerRef} controls style={{ display: transcription ? "block" : "none" }} />
    </div>
  );
};

export default AudioTranscription;
