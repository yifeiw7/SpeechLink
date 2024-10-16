import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer

# Set device and data type depending on whether CUDA (GPU) is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model identifier for Whisper large model
model_id = "openai/whisper-large-v3-turbo"

# Load the model with specific options for memory and device efficiency
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Load the processor for tokenizing and feature extraction
processor = AutoProcessor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition (ASR)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True  # Enable timestamps in the results
)

# Function to load transcription from text file
def load_transcript(txt_file):
    with open(txt_file, 'r') as file:
        return file.read().strip()

# Function to evaluate a pair of .wav and .txt files
def evaluate_wav_and_txt(wav_file, txt_file):
    # Load the ground truth transcription
    ground_truth = load_transcript(txt_file)

    # Transcribe the .wav file using Whisper
    result = pipe(wav_file)

    # Get the transcribed text
    predicted_transcript = result["text"].strip()

    # Calculate Word Error Rate (WER)
    error_rate = wer(ground_truth, predicted_transcript)

    return {
        "wav_file": wav_file,
        "ground_truth": ground_truth,
        "predicted": predicted_transcript,
        "WER": error_rate
    }

# Directory paths
wav_dir = "F/F01/Session1/wav_headMic"
txt_dir = "F/F01/Session1/prompts"

# List of .wav files to evaluate
wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

# Evaluate each file
for wav_file in wav_files:
    # Extract the base name (without extension) to find corresponding .txt file
    base_name = os.path.splitext(wav_file)[0]
    txt_file = os.path.join(txt_dir, f"{base_name}.txt")

    # Full path to the .wav file
    wav_file_path = os.path.join(wav_dir, wav_file)

    # Evaluate the pair of .wav and .txt
    result = evaluate_wav_and_txt(wav_file_path, txt_file)

    # Print or log the result
    print(f"Evaluating {wav_file}:")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Predicted: {result['predicted']}")
    print(f"Word Error Rate (WER): {result['WER']}\n")
