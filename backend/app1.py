import io
import traceback
from flask import Flask, request, jsonify
from pydub import AudioSegment  # For converting audio to WAV format
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Endpoint to handle audio file transcription. Accepts an audio file via POST request,
    processes it using the Whisper model, and returns the transcription as JSON.
    """
    try:
        # Check if a file was uploaded in the request
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        # Retrieve and log file details
        audio_file = request.files["file"]
        print(f"Received file: {audio_file.filename}, Content-Type: {audio_file.content_type}")

        # Convert the uploaded audio to WAV format using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        audio = audio.set_frame_rate(16000)  # Ensure it's resampled to 16 kHz
        audio = audio.set_channels(1)  # Ensure mono audio for compatibility

        # Export the audio to BytesIO for further processing
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        # Load audio data as a waveform array using librosa or soundfile
        import soundfile as sf
        wav_io.seek(0)
        audio_data, sampling_rate = sf.read(wav_io)

        # Process the audio waveform with the Whisper processor
        inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")

        # Generate transcription
        predicted_ids = model.generate(inputs.input_features)
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]  # Decodes the generated text

        # Return transcription result as JSON
        print({"transcript": transcript}) 
        return jsonify({"transcript": transcript})

    except Exception as e:
        print("Transcription error:", e)  # Log error details
        traceback.print_exc()  # Print the full stack trace for debugging
        return jsonify({"error": "Transcription failed", "details": str(e)}), 500

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)